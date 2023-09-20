import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP wrapper
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV
from src.classes.ClassPredictor import AncestralSamplingPredictor, EulerMaruyamaPredictor
from src.classes.ClassSDESampler import SDESampler
from src.classes.ClassTrainer import DiffusionModelTrainer
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.math_functions import chiSquared_test, reduce_to_fBn, compute_fBm_cov, permutation_test, \
    energy_statistic, MMD_statistic
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset, \
    plot_diffCov_heatmap, \
    plot_tSNE, plot_subplots
from torch.distributed.elastic.multiprocessing.errors import record

def ddp_setup(backend: str) -> None:
    """
    DDP setup to allow processes to discover and communicate with each other with TorchRun
    :param backend: Gloo vs NCCL for CPU vs GPU, respectively
    :return: None
    """
    #os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "1000"
    init_process_group(backend=backend)#, rank=rank, world_size=world_size)


def prepare_data(data: np.ndarray, batch_size: int, config: ConfigDict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data into train, eval, test sets and create DataLoaders for training
        :param data: Training data
        :param batch_size: Batch size
        :param config: ML Collection dictionary
        :return: Train, Validation, Test dataloaders
    """
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    # TODO: Do I want a Distributed Sampler for validation and test too?
    # TODO: Shuffle is turned to False when using a Sampler, since it specifies the shuffling strategy
    # TODO: sampler=DistributedSampler(train)
    if config.has_cuda:
        trainLoader, valLoader, testLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                        sampler=DistributedSampler(train)), \
                                             DataLoader(val, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                        sampler=DistributedSampler(val)), \
                                             DataLoader(test, batch_size=batch_size, pin_memory=True, shuffle=False,
                                                        sampler=DistributedSampler(test))
    else:
        trainLoader, valLoader, testLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=True,
                                                        num_workers=0), \
                                             DataLoader(val, batch_size=batch_size, pin_memory=True, shuffle=True,
                                                        num_workers=0), \
                                             DataLoader(test, batch_size=batch_size, pin_memory=True, shuffle=True,
                                                        num_workers=0)

    return trainLoader, valLoader, testLoader

@record
def initialise_training(data: np.ndarray, config: ConfigDict,
                        diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                        scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching]) -> None:
    """
    Helper function to initiate training
        :param data: Dataset
        :param config: Configuration dictionary with relevant parameters
        :param diffusion: SDE model
        :param scoreModel: Score network architecture
        :return: None
    """
    if config.has_cuda:
        # Spawn one process for every GPU device, automatically assign rank
        #world_size = torch.cuda.device_count()
        #mp.spawn(train_and_save_diffusion_model, args=(world_size, data, config, diffusion, scoreModel),
                 #nprocs=world_size)
        train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
    else:
        # Only one CPU device, so we have 1 core on 1 device
        train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)

def train_and_save_diffusion_model(data: np.ndarray,
                                   config: ConfigDict,
                                   diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                                   scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching]) -> None:
    """
    Helper function to initiate training
        :param rank: Unique process indentifier
        :param world_size: Total number of processes
        :param data: Dataset
        :param config: Configuration dictionary with relevant parameters
        :param diffusion: SDE model
        :param scoreModel: Score network architecture
        :return: None
    """
    if config.has_cuda:
        ddp_setup(backend="nccl")#, rank=rank, world_size=world_size)
        device = int(os.environ["LOCAL_RANK"]) # TODO: Use device = rank passed by mp.spawn or COMPLETELY by pass with torchrun and int[os.environ["LOCAL_RANK"]] OR is the local path environ the same when we call Trainer (should be!)?
    else:
        ddp_setup(backend="gloo")#, rank=rank, world_size=world_size)
        #torch.set_num_threads(int(0.75 * os.cpu_count()))
        device = torch.device("cpu")

    # Preprocess data
    trainLoader, valLoader, testLoader = prepare_data(data=data, batch_size=config.batch_size, config=config)

    # Define optimiser
    optimiser = torch.optim.Adam((scoreModel.parameters()), lr=config.lr)  # TODO: Do we not need DDP?

    # Define trainer
    train_eps, end_diff_time, max_diff_steps, checkpoint_freq = config.train_eps, config.end_diff_time, config.max_diff_steps, config.save_freq
    
    # TODO: When using DDP, set device = rank passed by mp.spawn OR by torchrun
    trainer = DiffusionModelTrainer(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                    checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                    loss_aggregator=torchmetrics.aggregation.MeanMetric,
                                    snapshot_path=config.snapshot_path, device=device,
                                    train_eps=train_eps,
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps)

    # Start training
    trainer.train(max_epochs=config.max_epochs, model_filename=config.filename)

    # Cleanly exit the DDP training
    """destroy_process_group()"""

@record
def reverse_sampling(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                     scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], data_shape: Tuple[int, int],
                     config: ConfigDict) -> torch.Tensor:
    """
    Helper function to initiate sampling
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    # TODO: DDP cannot be used here since sampling is sequential, so only single-machine, single-GPU/CPU?

    if config.has_cuda:
        device = 0
    else:
        torch.set_num_threads(int(os.cpu_count()))
        device = torch.device("cpu")
    
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device]
    predictor = AncestralSamplingPredictor(
        *predictor_params) if config.predictor_model == "ancestral" else EulerMaruyamaPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, torch.Tensor([config.snr]), device, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = SDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor, corrector=corrector)

    # Sample
    final_samples = sampler.sample(shape=data_shape, torch_device=device)
    return final_samples  # TODO Check if need to detach


def check_convergence_at_diffTime(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
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


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator, config:ConfigDict) -> None:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples of fractional Brownian motion (or its increments)
        :param generated_samples: Final reverse-time diffusion samples
        :param h: Hurst index
        :param rng: Default random number generator
        :param config: Configuration dictionary for experiment
        :return: None
    """
    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim, isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)

    plot_diffCov_heatmap(expec_cov, gen_cov, annot=config.annot, image_path=config.image_path+"_diffCov")
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst, samples=reduce_to_fBn(true_samples, reduce=config.isfBm), isUnitInterval=config.unitInterval)
    print("Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                     c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst, samples=reduce_to_fBn(generated_samples, reduce=config.isfBm), isUnitInterval=config.unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_tSNE(x=true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"],  image_path=config.image_path+"_tSNE") \
        if config.timeDim > 2 else plot_dataset(true_samples, generated_samples,  image_path=config.image_path+"_scatter")
    if config.eval_marginals: plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=config.timeDim,  image_path=config.image_path)
    if config.permute_test:
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


def evaluate_circle_performance(true_samples: np.ndarray, generated_samples: np.ndarray, config:ConfigDict) -> None:
    """
    Compute various quantitative and qualitative metrics on final reverse-time diffusion samples for circle dataset
        :param true_samples: Exact samples from circle dataset
        :param generated_samples: Final reverse-time diffusion samples
        :param config: Configuration dictionary for experiment
        :return: None
    """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)

    plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter")

    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=gen_cov, image_path=config.image_path + "_scatter")
    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=2, image_path=config.image_path)

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
