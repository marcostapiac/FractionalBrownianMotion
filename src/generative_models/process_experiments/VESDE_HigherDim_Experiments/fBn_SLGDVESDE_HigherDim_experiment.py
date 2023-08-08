import pickle

import numpy as np
import torch

from src.classes import ClassVESDEDiffusion
from src.classes.ClassVESDEDiffusion import VESDEDiffusion
from src.classes.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import generate_fBn, chiSquared_test
from utils.plotting_functions import plot_dataset, plot_diffusion_marginals

LR = 1e-4
NUM_EPOCHS = 400
BATCH_SIZE = 256


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

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

    plot_dataset(true_samples, generated_samples)

    # Visualise data
    # plot_tSNE(true_samples, generated_samples, ["True fBn Samples", "Generated fBn Samples"])

    # Permutation test for kernel statistic
    # print("MMD Permutation test: p-value {}".format(
    #    permutation_test(true_samples, generated_samples, compute_statistic=MMD_statistic, num_permutations=1000)))
    # Permutation test for energy statistic
    # print("Energy Permutation test: p-value {}".format(
    #    permutation_test((true_samples-np.mean(true_samples, axis=0))/np.std(true_samples, axis=0), (generated_samples-np.mean(generated_samples, axis=0))/np.std(generated_samples, axis=0), compute_statistic=energy_statistic, num_permutations=1000)))


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVESDEDiffusion,
                   rng: np.random.Generator, data: np.ndarray, sampleEps: float) -> None:
    assert (0. < hurst <= 1.)
    """ Perform ancestral sampling given model """
    fBn_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0, data=data,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps),
                                            sigNoiseRatio=0.1)
    true_samples = generate_fBn(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBn_samples)


if __name__ == "__main__":
    h, td = 0.7, 1024
    numSamples = 100000
    trainEps = 1e-3
    sampleEps = 1e-3
    N = 100
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/a_hundredthousand_fBn_samples_H07_T{}.npy".format(td))[:numSamples // 5,
               :]
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_fBn_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
            scoreModel = TimeSeriesNoiseMatching()
            diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                       noiseFactor=2.)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBn_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(T=td, S=numSamples, H=h, rng=rng)
        np.save(config.ROOT_DIR + "data/a_hundredthousand_fBn_samples_H07_T{}.npy".format(td), data)
        data = data[:numSamples // 5, :]
        scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                   noiseFactor=2.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBn_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 1000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
