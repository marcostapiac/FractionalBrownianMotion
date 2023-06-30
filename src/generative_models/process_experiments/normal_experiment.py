import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from scipy.stats import kstest
from tqdm import tqdm

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import permutation_test, MMD_statistic, energy_statistic
from utils.plotting_functions import qqplot

LR = 1e-3
NUM_EPOCHS = 200
BATCH_SIZE = 64


def generate_norm(T: int, S: int, rng: np.random.Generator):
    return np.array([rng.normal(size=T) for _ in range(S)]).reshape(S, T)


def chiSquared_test(T: int, samples: np.ndarray):
    def chiSquared(normal_samples):
        return np.sum([i ** 2 for i in normal_samples])

    alpha = 0.05
    S = samples.shape[0]
    critUpp = chi2.ppf(q=1. - 0.5 * alpha, df=S * T - 1)  # Upper alpha quantile, and dOf = T - 1
    critLow = chi2.ppf(q=0.5 * alpha, df=S * T - 1)  # Lower alpha quantile, and d0f = T -1
    ts = []
    for i in tqdm(range(S)):
        tss = chiSquared(samples[i, :])
        ts.append(tss)
    return critLow, np.sum(ts), critUpp


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, T: int) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """
    # Marginal test plots
    ts = np.random.randint(low=1, high=T, size=3)
    for t in ts:
        true_norm_t = true_samples[:, t].flatten()
        generated_norm_t = generated_samples[:, t].flatten()
        qqplot(x=(true_norm_t - np.mean(true_norm_t)) / np.std(true_norm_t),
               y=(generated_norm_t - np.mean(generated_norm_t)) / np.std(generated_norm_t), xlabel="True Samples",
               ylabel="Generated Samples", plottitle="Marginal Q-Q Plot at random time {}".format(t), log=False)
        print(kstest((true_norm_t - np.mean(true_norm_t)) / np.std(true_norm_t),
                     (generated_norm_t - np.mean(generated_norm_t)) / np.std(generated_norm_t)))
        plt.show()
        plt.close()

    # Chi-2 test for joint distribution
    c2 = chiSquared_test(T=T, samples=generated_samples)
    print("Chi-Squared test: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1], c2[2]))

    # Permutation test for kernel statistic
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples, generated_samples, compute_statistic=MMD_statistic, num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples, generated_samples, compute_statistic=energy_statistic, num_permutations=1000)))


def run_experiment(timeDim: int, dataSize: int, diffusion: DenoisingDiffusion,
                   rng: np.random.Generator) -> None:
    """ Perform ancestral sampling given model """
    norm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim)
    true_samples = generate_norm(T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, norm_samples, T=timeDim)


if __name__ == "__main__":
    td = 64
    rng = np.random.default_rng()
    try:
        data = np.load("../../data/a_million_norm_samples_H07_T{}.npy".format(td))
        data = np.cumsum(data, axis=1)[:10000, :]
        try:
            file = open("models/trained_norm_diffusion_model_T{}".format(td), 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            N = 2000
            model = save_and_train_diffusion_model(data, numDiffSteps=N,
                                                   model_filename="models/trained_norm_diffusion_model_T{}".format(td),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, rng=rng)
    except FileNotFoundError:
        numSamples = 1000000
        data = generate_norm(T=td, S=numSamples, rng=rng)
        np.save("../../data/a_million_norm_samples_H07_T{}.npy".format(td), data)
        data = np.cumsum(data, axis=1)[:10000, :]
        N = 2000
        model = save_and_train_diffusion_model(data, numDiffSteps=N,
                                               model_filename="models/trained_norm_diffusion_model_T{}".format(td),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, rng=rng)
    run_experiment(diffusion=model, timeDim=td, rng=rng, dataSize=100)
