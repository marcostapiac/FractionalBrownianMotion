import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.generative_modelling.models import ClassOUDiffusion
from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils import project_config
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import generate_circles
from utils.plotting_functions import plot_final_diffusion_marginals

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 64


def observation_model(latent: np.ndarray) -> np.ndarray:
    return latent + 6. * np.ones_like(latent) + np.sqrt(0.1) * np.random.randn(*latent.shape)


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """

    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.3, label="Generated Samples")
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


def run_experiment(observed: np.ndarray, timeDim: int, dataSize: int, latent: np.ndarray, diffusion: ClassOUDiffusion,
                   sampleEps) -> None:
    """ Perform ancestral sampling given model """
    latent_samples = diffusion.conditional_reverse_process(observations=observed[:dataSize],
                                                           latent=torch.from_numpy(latent).to(torch.float32),
                                                           dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                                           reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                                       steps=diffusion.numDiffSteps))
    true_samples = generate_circles(T=timeDim, S=dataSize)
    evaluate_performance(latent_samples, true_samples)


if __name__ == "__main__":
    td = 2
    N = 1000
    numSamples = 1000000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        latent = np.load(config.ROOT_DIR + "data/a_million_circle_samples_T{}.npy".format(td))[:numSamples // 50, :]
        observations = torch.from_numpy(observation_model(latent)).to(torch.float32)
        try:
            file = open(
                config.ROOT_DIR + "src/generative_modelling/trained_models/trained_circle_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = TimeSeriesScoreMatching()
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
            model = save_and_train_diffusion_model(latent,
                                                   model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_circle_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        latent = generate_circles(T=td, S=numSamples, noise=0.03)
        np.save(config.ROOT_DIR + "data/a_million_circle_samples_T{}.npy".format(td), latent)
        latent = latent[:numSamples // 50, :]
        observations = observation_model(latent)
        scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32], max_diff_steps=N)
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
        model = save_and_train_diffusion_model(latent,
                                               model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_circle_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, timeDim=td, dataSize=2000, observed=observations, latent=latent,
                   sampleEps=sampleEps)
