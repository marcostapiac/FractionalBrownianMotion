import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.classes import ClassOUDiffusion
from src.classes.ClassNaiveMLP import NaiveMLP
from src.classes.ClassOUDiffusion import OUDiffusion
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import chiSquared_test, generate_fBn
from utils.plotting_functions import plot_diffusion_marginals

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 64


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    print(np.mean(true_samples, axis=0))
    print(np.mean(generated_samples, axis=0))
    print([0., 0.])
    print(np.cov(true_samples.T))
    print(np.cov(generated_samples.T))
    print([[1., 0.5 * 2 ** 1.4 - 1.], [0.5 * (2 ** 1.4) - 1., 1.]])

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(np.cov(true_samples.T)[0, 0],
                                                                     np.cov(true_samples.T)[0, 1]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(np.cov(generated_samples.T)[0, 0],
                                                                      np.cov(generated_samples.T)[0, 1]))
    print("Expected :: Var {} :: Covar {}".format(1., 0.5 * 2 ** 1.4 - 1.))
    assert (np.cov(generated_samples.T)[0, 1] == np.cov(generated_samples.T)[1, 0])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=generated_samples)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Time Dim 1")
    ax.set_ylabel("Time Dim 2")
    plt.legend()
    plt.show()

    # Visualise data
    # plot_tSNE(true_samples, generated_samples, ["True fBn Samples", "Generated fBn Samples"])
    # plot_tSNE(forward_prior_samples, reverse_prior_samples, ["True Gaussian Samples", "Generated Gaussian Samples"])

    # Permutation test for kernel statistic
    # print("MMD Permutation test: p-value {}".format(
    #    permutation_test(true_samples, generated_samples, compute_statistic=MMD_statistic, num_permutations=1000)))
    # Permutation test for energy statistic
    # print("Energy Permutation test: p-value {}".format(
    #    permutation_test((true_samples-np.mean(true_samples, axis=0))/np.std(true_samples, axis=0), (generated_samples-np.mean(generated_samples, axis=0))/np.std(generated_samples, axis=0), compute_statistic=energy_statistic, num_permutations=1000)))


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassOUDiffusion,
                   rng: np.random.Generator) -> None:
    assert (0. < hurst <= 1.)
    """ Perform ancestral sampling given model """
    fBn_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                            reverseTimes=torch.linspace(start=1., end=1e-5,
                                                                        steps=diffusion.numDiffSteps))
    true_samples = generate_fBn(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBn_samples)


if __name__ == "__main__":
    h, td = 0.7, 2
    N = 1000
    numSamples = 1000000
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/a_million_fBn_samples_H07_T{}.npy".format(td))[:numSamples // 50, :]
        try:
            file = open(config.ROOT_DIR + "src/generative_models/models/trained_fBn_OU_model_T{}_Ndiff{}".format(td, N),
                        'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            #scoreModel = TimeSeriesNoiseMatching()
            scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=1e-5)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBn_OU_model_T{}_Ndiff{}".format(
                                                       td, N),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/a_million_fBn_samples_H07_T{}.npy".format(td), data)
        data = data[:numSamples // 50, :]
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=1e-5)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBn_OU_model_T{}_Ndiff{}".format(
                                                   td, N),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=10000)
