import pickle
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV
from src.classes.ClassPredictor import AncestralSamplingPredictor, EulerMaruyamaPredictor
from src.classes.ClassSDESampler import SDESampler
from src.classes.ClassTrainer import DiffusionModelTrainer
from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.math_functions import chiSquared_test, reduce_to_fBn, compute_fBm_cov, permutation_test, \
    energy_statistic, MMD_statistic
from utils.plotting_functions import plot_and_save_loss_epochs, plot_final_diffusion_marginals, plot_dataset, \
    plot_diffCov_heatmap, \
    plot_tSNE, plot_subplots


def prepare_data(data: np.ndarray, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data into train, eval, test sets and create DataLoaders for training
        :param data: Training data
        :param batch_size: Batch size
        :return: Train, Validation, Test dataloaders
    """
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    trainLoader, valLoader, testLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=True), \
                                         DataLoader(val, batch_size=batch_size, pin_memory=True, shuffle=True), \
                                         DataLoader(test, batch_size=batch_size, pin_memory=True,
                                                    shuffle=True)  # Returns iterator
    return trainLoader, valLoader, testLoader


def train_diffusion_model(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUDiffusion],
                          trainLoader: torch.utils.data.DataLoader,
                          valLoader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer, nEpochs: int) -> Tuple[
    np.array, np.array]:
    """
    Abstract function to train model
        :param diffusion: Untrained model
        :param trainLoader: Training data DataLoader
        :param valLoader: Validation data DataLoader
        :param opt: Optimiser
        :param nEpochs: Number of training epochs
        :return: Train and Validation Losses
    """
    # TODO: Need to move variables to correct device -- but which ones?
    train_losses = []
    val_losses = []
    for i in range(nEpochs):
        train_loss = diffusion.one_epoch_diffusion_train(trainLoader=trainLoader, opt=opt)
        val_loss = diffusion.evaluate_diffusion_model(loader=valLoader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            "Percent Completed {:0.4f} Train :: Val Losses, {:0.4f} :: {:0.4f}".format((i + 1) / nEpochs, train_loss,
                                                                                       val_loss))
    return train_losses, val_losses


def save_and_train_diffusion_model(data: np.ndarray, model_filename: str, batch_size: int,
                                   nEpochs: int, lr: float,
                                   diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUDiffusion]) -> Union[
    VPSDEDiffusion, VESDEDiffusion, OUDiffusion]:
    """
    Abstract function which calls training function, plots losses, and saves trained model
        :param data: Training dataset
        :param model_filename: Filename to store trained model
        :param batch_size: Size of batch during training
        :param nEpochs: Number of training epochs
        :param lr: Learning rate for optimiser
        :param diffusion: Untrained diffusion model
        :return: Trained diffusion model
    """
    trainLoader, valLoader, testLoader = prepare_data(data, batch_size=batch_size)

    # Set up optimiser
    optimiser = torch.optim.Adam((diffusion.parameters()), lr=lr)  # No need to move to device

    # Compute number of parameters
    params = 0
    for item in optimiser.param_groups[0]["params"]:
        params += np.prod(item.shape)
    print("Number of model parameters : {}".format(params))

    # Call training function
    train_loss, val_loss = train_diffusion_model(diffusion=diffusion, trainLoader=trainLoader,
                                                 valLoader=valLoader,
                                                 opt=optimiser, nEpochs=nEpochs)
    # Plot loss curves
    plot_and_save_loss_epochs(epochs=np.arange(1, nEpochs + 1, step=1), val_loss=val_loss,
                              train_loss=np.array(train_loss))
    # Save trained model to pickle file
    file = open(model_filename, "wb")
    pickle.dump(diffusion, file)
    file.close()
    return diffusion


def revamped_train_and_save_diffusion_model(data: np.ndarray, model_filename: str, batch_size: int,
                                            nEpochs: int, lr: float, train_eps: float, end_diff_time: float,
                                            max_diff_steps: int,
                                            diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUDiffusion],
                                            scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                                            checkpoint_freq: int) -> Union[NaiveMLP, TimeSeriesScoreMatching]:
    trainLoader, valLoader, testLoader = prepare_data(data, batch_size=batch_size)
    optimiser = torch.optim.Adam((scoreModel.parameters()), lr=lr)  # No need to move to device
    trainer = DiffusionModelTrainer(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                    checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                    loss_aggregator=torchmetrics.aggregation.MeanMetric, gpu_id=0, train_eps=train_eps,
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps)
    trainer.train(max_epochs=nEpochs, model_filename=model_filename)
    return scoreModel


def reverse_sampling(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUDiffusion],
                     scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], data_shape: Tuple[int, int], config:ConfigDict) -> torch.Tensor:
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps]
    predictor = AncestralSamplingPredictor(*predictor_params) if config.predictor_model == "ancestral" else EulerMaruyamaPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, config.snr, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = SDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor, corrector=corrector)

    # Sample
    final_samples = sampler.sample(shape=data_shape)
    return final_samples  # TODO Check if need to detach


def check_convergence_at_diffTime(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUDiffusion],
                                  t: int, dataSamples: np.ndarray) -> Tuple[np.ndarray,
                                                                            np.ndarray,
                                                                            list[str]]:
    """
    Function generates forward process and backward process samples from the same diffusion time index
        :param diffusion: Trained diffusion model
        :param t: Diffusion time index
        :param dataSamples: Exact samples from $p_{data}$
        :return: (Forward process samples, Backward process samples, Labels for each)
    """
    forward_samples_at_t, _ = diffusion.forward_process(dataSamples=torch.from_numpy(dataSamples),
                                                        diffusionTimes=torch.ones(dataSamples.shape[0],
                                                                                  dtype=torch.long) * (
                                                                           torch.from_numpy(np.array([t]))))
    # Generate backward samples
    backward_samples_at_t = diffusion.reverse_process(dataSize=forward_samples_at_t.shape[0],
                                                      timeDim=forward_samples_at_t.shape[1], timeLim=t + 1)
    labels = ["Forward Samples at time {}".format(t + 1), "Backward Samples at time {}".format(t + 1)]
    return forward_samples_at_t.numpy(), backward_samples_at_t, labels


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, h: float, td: int,
                         rng: np.random.Generator, unitInterval: bool, annot: bool, evalMarginals: bool,
                         isfBm: bool, permute_test: bool) -> None:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples of fractional Brownian motion (or its increments)
        :param generated_samples: Final reverse-time diffusion samples
        :param h: Hurst index
        :param td: Length of timeseries / dimensionality of each datapoint
        :param rng: Default random number generator
        :param unitInterval: Indicates whether fBm was simulated on [0,1] or [0, td]
        :param annot: Indicates whether to annotate covariance plot
        :param evalMarginals: Indicates whether to plot marginal Q-Q plots and compute associated KS statistic
        :param isfBm: Indicates whether samples are fBm or its increments
        :param permute_test: Indicates whether to conduct permutation tes
        :return: None
    """
    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=h, rng=rng), T=td, isUnitInterval=unitInterval)
    print("Expected Covariance :: ", expec_cov)

    plot_diffCov_heatmap(expec_cov, gen_cov, annot=annot)
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=reduce_to_fBn(true_samples, reduce=isfBm), isUnitInterval=unitInterval)
    print("Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                     c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=reduce_to_fBn(generated_samples, reduce=isfBm), isUnitInterval=unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_tSNE(true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"]) \
        if td > 2 else plot_dataset(true_samples, generated_samples)
    if evalMarginals: plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)
    if permute_test:
        # Permutation test for kernel statistic
        test_L = min(2000, true_samples.shape[0])
        print("MMD Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                             num_permutations=1000)))
        # Permutation test for energy statistic
        print("Energy Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                             num_permutations=1000)))


def compute_circle_proportions(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """
    Function computes approximate ratio of samples in inner vs outer circle of circle dataset
        :param true_samples: data samples exactly using sklearn's "make_circle" function
        :param generated_samples: final reverse-time diffusion samples
        :return: None
    """
    innerb = 0
    outerb = 0
    innerf = 0
    outerf = 0
    S = true_samples.shape[0]
    for i in range(S):
        bkwd = generated_samples[i]
        fwd = true_samples[i]
        rb = np.sqrt(bkwd[0] ** 2 + bkwd[1] ** 2)
        rf = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
        if rb <= 2.1:
            innerb += 1
        elif 3.9 <= rb:
            outerb += 1
        if rf <= 2.1:
            innerf += 1
        elif 3.9 <= rf:
            outerf += 1

    print("Generated: Inner {} vs Outer {}".format(innerb / S, outerb / S))
    print("True: Inner {} vs Outer {}".format(innerf / S, outerf / S))


def evaluate_circle_performance(true_samples: np.ndarray, generated_samples: np.ndarray, td: int) -> None:
    """
    Compute various quantitative and qualitative metrics on final reverse-time diffusion samples for circle dataset
        :param true_samples: Exact samples from circle dataset
        :param generated_samples: Final reverse-time diffusion samples
        :param td: Dimension of each sample
        :return: None
    """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)

    plot_dataset(true_samples, generated_samples)

    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=gen_cov)
    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)

    compute_circle_proportions(true_samples, generated_samples)

    # Permutation test for kernel statistic
    test_L = min(2000, true_samples.shape[0])
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                         num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                         num_permutations=1000)))


def compare_fBm_to_approximate_fBm(generated_samples: np.ndarray, h: float, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing final reverse-time diffusion fBm samples to approximate fBm samples from approximate simulations
        :param generated_samples: Exact fBm samples
        :param h: Hurst index
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    generator = FractionalBrownianNoise(H=h, rng=rng)
    S = min(20000, generated_samples.shape[0])
    approx_samples = np.empty((S, td))
    for _ in range(S):
        approx_samples[_, :] = generator.paxon_simulation(
            N_samples=td).cumsum()  # TODO: Are we including initial sample?
    plot_tSNE(generated_samples, y=approx_samples,
              labels=["Reverse Diffusion Samples", "Approximate Samples: Paxon Method"])


def compare_fBm_to_normal(generated_samples: np.ndarray, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing reverse-time diffusion samples to standard normal samples
        :param generated_samples: Exact fBm samples
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    S = min(20000, generated_samples.shape[0])
    normal_rvs = np.empty((S, td))
    for _ in range(S):
        normal_rvs[_, :] = rng.standard_normal(td)
    plot_tSNE(generated_samples, y=normal_rvs, labels=["Reverse Diffusion Samples", "Standard Normal RVS"])


def gen_and_store_statespace_data(Xs=None, Us=None, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11,
                                  T=1e-3 * 2 ** 11):
    """
    Generate observation and latent signal from CEV model and store it as pickle file
        :param Xs: Optional parameter, array containing latent signal process
        :param Us: Optional parameter, array containing observation process
        :param muU: Drift parameter for observation process
        :param muX: Drift parameter for latent process
        :param gamma: Mean reversion parameter for latent process
        :param X0: Initial value for latent process
        :param U0: Initial value for observation process
        :param H: Hurst Index
        :param N: Length of time series
        :param T: Terminal simualtion time for a process on [0, T]
        :return: None
    """
    assert ((Xs is None and Us is None) or (Xs is not None and Us is not None))
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    if Xs is None:
        m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, alpha=alpha, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    df = pd.DataFrame.from_dict(data={'Log-Price': Us, 'Volatility': Xs})
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), np.array([Xs, Us]), np.array([None, None]),
                  np.array(["Time", "Time"]),
                  np.array(["Volatility", "Log Price"]),
                  "Project Model Simulation")
    df.to_csv('../data/raw_data_simpleObsModel_{}_{}.csv'.format(int(np.log2(N)), int(10 * H)), index=False)
