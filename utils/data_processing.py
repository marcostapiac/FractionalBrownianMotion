import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.classes import ClassOUDiffusion, ClassVESDEDiffusion, ClassVPSDEDiffusion
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_fBn_cov, chiSquared_test, permutation_test, MMD_statistic, energy_statistic, \
    fBm_to_fBn, compute_fBm_cov
from utils.plotting_functions import plot_loss_epochs, plot_diffusion_marginals, plot_dataset, plot_diffCov_heatmap, \
    plot_tSNE


def prepare_data(data: np.ndarray, batch_size: int) -> [DataLoader, DataLoader, DataLoader]:
    S, T = data.shape
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    train, val, test = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    trainLoader, valLoader, testLoader = DataLoader(train, batch_size=batch_size, shuffle=True), \
                                         DataLoader(val, batch_size=batch_size, shuffle=True), \
                                         DataLoader(test, batch_size=batch_size, shuffle=True)  # Returns iterator
    return trainLoader, valLoader, testLoader


def train_diffusion_model(diffusion: type[ClassVPSDEDiffusion, ClassVESDEDiffusion, ClassOUDiffusion],
                          trainLoader: torch.utils.data.DataLoader,
                          valLoader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer, nEpochs: int) -> [
    np.array, np.array]:
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
                                   diffusion: type[ClassVPSDEDiffusion, ClassVESDEDiffusion, ClassOUDiffusion]) -> \
        type[ClassVPSDEDiffusion, ClassVESDEDiffusion, ClassOUDiffusion]:
    """ Prepare data for training """
    trainLoader, valLoader, testLoader = prepare_data(data, batch_size=batch_size)

    """ Prepare optimiser """
    optimiser = torch.optim.Adam((diffusion.parameters()), lr=lr)  # No need to move to device
    params = 0
    for item in optimiser.param_groups[0]["params"]:
        params += np.prod(item.shape)
    print("Number of model parameters : {}".format(params))
    """ Training """
    train_loss, val_loss = train_diffusion_model(diffusion=diffusion, trainLoader=trainLoader,
                                                 valLoader=valLoader,
                                                 opt=optimiser, nEpochs=nEpochs)
    plot_loss_epochs(epochs=np.arange(1, nEpochs + 1, step=1), val_loss=val_loss, train_loss=np.array(train_loss))
    """ Save model """
    file = open(model_filename, "wb")
    pickle.dump(diffusion, file)
    file.close()
    return diffusion


def check_convergence_at_diffTime(diffusion: type[ClassVPSDEDiffusion, ClassVESDEDiffusion, ClassOUDiffusion],
                                  t: int, dataSamples: np.ndarray) -> [np.ndarray,
                                                                       np.ndarray,
                                                                       list[str]]:
    forward_samples_at_t, _ = diffusion.forward_process(dataSamples=torch.from_numpy(dataSamples),
                                                        diffusionTimes=torch.ones(dataSamples.shape[0],
                                                                                  dtype=torch.long) * (
                                                                           torch.from_numpy(np.array([t]))))
    # Generate backward samples
    backward_samples_at_t = diffusion.reverse_process(dataSize=forward_samples_at_t.shape[0],
                                                      timeDim=forward_samples_at_t.shape[1], timeLim=t + 1)
    labels = ["Forward Samples at time {}".format(t + 1), "Backward Samples at time {}".format(t + 1)]
    return forward_samples_at_t.numpy(), backward_samples_at_t, labels


def evaluate_fBn_performance(true_samples: np.ndarray, generated_samples: np.ndarray, h: float, td: int,
                             rng: np.random.Generator, unitInterval: bool) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    expec_mean = np.array([0., 0.])
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(*expec_mean))
    print("Original Data :: \n [[{}, {}]\n[{},{}]]".format(*np.cov(true_samples, rowvar=False).flatten()))
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: \n [[{}, {}]\n[{},{}]]".format(*gen_cov.flatten()))
    expec_cov = compute_fBn_cov(FractionalBrownianNoise(H=h, rng=rng), td=td, isUnitInterval=unitInterval)
    print("Expected :: \n [[{}, {}]\n[{},{}]]".format(*expec_cov.flatten()))

    assert (np.cov(generated_samples.T)[0, 1] == np.cov(generated_samples.T)[1, 0])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=true_samples, isUnitInterval=unitInterval)
    print("Chi-Squared test for original: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                         c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=generated_samples, isUnitInterval=unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    plot_diffCov_heatmap(expec_cov, gen_cov)

    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    plot_dataset(true_samples, generated_samples)
    plot_tSNE(true_samples, y=generated_samples, labels=["Original", "Generated"])
    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    """
    # Permutation test for kernel statistic
    test_L = min(2000, true_samples.shape[0])
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                         num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                         num_permutations=1000)))
    """


def evaluate_fBm_performance(true_samples: np.ndarray, generated_samples: np.ndarray, h: float, td: int,
                             rng: np.random.Generator, unitInterval: bool, annot: bool, evalMarginals: bool) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=h, rng=rng), td=td, isUnitInterval=unitInterval)
    print("Expected Covariance :: ", expec_cov)

    plot_diffCov_heatmap(expec_cov, gen_cov, annot=annot)
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=fBm_to_fBn(true_samples), isUnitInterval=unitInterval)
    print("Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                     c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=fBm_to_fBn(generated_samples), isUnitInterval=unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_tSNE(true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"]) \
        if td > 2 else plot_dataset(true_samples, generated_samples)
    if evalMarginals: plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    """
    test_L = min(2000, true_samples.shape[0])
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                         num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                         num_permutations=1000)))
    """

def compute_circle_proportions(true_samples:np.ndarray, generated_samples:np.ndarray)->None:
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
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)

    plot_dataset(true_samples, generated_samples)

    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    compute_circle_proportions(true_samples, generated_samples)



    # Permutation test for kernel statistic
    # test_L = min(2000, true_samples.shape[0])
    # print("MMD Permutation test: p-value {}".format(
    #    permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
    #                     num_permutations=1000)))
    # Permutation test for energy statistic
    # print("Energy Permutation test: p-value {}".format(
    #    permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
    #                     num_permutations=1000)))

def evaluate_SDE_performance(true_samples: np.ndarray, generated_samples: np.ndarray, td: int) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)

    plot_diffCov_heatmap(true_cov, gen_cov, annot=False)

    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]
    plot_tSNE(true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"])

    """
    test_L = min(2000, true_samples.shape[0])
    true_samples, generated_samples = true_samples[:test_L], generated_samples[:test_L]
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples, generated_samples, compute_statistic=MMD_statistic,
                         num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples, generated_samples, compute_statistic=energy_statistic,
                         num_permutations=1000)))
    """


def compare_fBm_to_approximate_fBm(generated_samples: np.ndarray, h: float, td: int, rng: np.random.Generator) -> None:
    generator = FractionalBrownianNoise(H=h, rng=rng)
    S = min(20000, generated_samples.shape[0])
    approx_samples = np.empty((S, td))
    for _ in range(S):
        approx_samples[_, :] = generator.paxon_simulation(N_samples=td).cumsum()
    plot_tSNE(generated_samples, y=approx_samples,
              labels=["Reverse Diffusion Samples", "Approximate Samples: Paxon Method"])


def compare_fBm_to_normal(generated_samples: np.ndarray, td: int, rng: np.random.Generator) -> None:
    S = min(20000, generated_samples.shape[0])
    normal_rvs = np.empty((S, td))
    for _ in range(S):
        normal_rvs[_, :] = rng.standard_normal(td)
    plot_tSNE(generated_samples, y=normal_rvs, labels=["Reverse Diffusion Samples", "Standard Normal RVS"])
