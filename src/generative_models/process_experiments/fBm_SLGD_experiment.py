import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2
from scipy.stats import kstest
from tqdm import tqdm

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassSLGDDiffusion import ScoreBasedDiffusion
from src.classes.ClassScoreBasedNoiseMatching import ScoreBasedNoiseMatching
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils.data_processing import save_and_train_diffusion_model
from utils.plotting_functions import qqplot, plot_tSNE

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 256


def generate_fBn(H: float, T: int, S: int, rng: np.random.Generator) -> np.array:
    generator = FractionalBrownianNoise(H=H, rng=rng)
    data = np.empty(shape=(S, T))
    for i in tqdm(range(S)):
        data[i, :] = generator.circulant_simulation(N_samples=T)
    return data


def generate_fBm(H: float, T: int, S: int, rng: np.random.Generator) -> np.array:
    data = generate_fBn(H=H, T=T, S=S, rng=rng)
    return np.cumsum(data, axis=1)


def chiSquared_test(T: int, H: float, samples: np.ndarray):
    def standardise_sample(fBn_sample, invL):
        return np.squeeze(invL @ fBn_sample)

    def chiSquared(fBn_sample, invL):
        standard_sample = standardise_sample(fBn_sample, invL)
        return np.sum([i ** 2 for i in standard_sample])

    fbn = FractionalBrownianNoise(H)
    invL = np.linalg.inv(
        np.linalg.cholesky(np.atleast_2d(
            [[fbn.covariance(i - j) for j in range(T)] for i in tqdm(range(T))])))
    alpha = 0.05
    crit = chi2.ppf(q=1 - alpha, df=T - 1)  # Upper alpha quantile, and dOf = N - 1
    M = samples.shape[0]
    ts = []
    for i in tqdm(range(M)):
        tss = chiSquared(samples[i, :], invL)
        ts.append(tss)
    return np.mean(ts), crit


def fBm_to_fBn(fBm_timeseries: np.ndarray) -> np.array:
    T = fBm_timeseries.shape[1]
    return fBm_timeseries - np.insert(fBm_timeseries[:, :T - 1], 0, 0., axis=1)


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, T: int, H: float,
                         rng: np.random.Generator, diffusion: DenoisingDiffusion) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """
    # Marginal test plots
    for t in np.arange(start=0, stop=T, step=1):
        true_fBm_t = true_samples[:, t].flatten()
        generated_fBm_t = generated_samples[:, t].flatten()
        qqplot(x=true_fBm_t,
               y=generated_fBm_t, xlabel="True Samples",
               ylabel="Generated Samples", plottitle="Marginal Q-Q Plot at time {}".format(t + 1), log=False)
        print(kstest(true_fBm_t, generated_fBm_t))
        plt.show()
        plt.close()

    # Check convergence to target
    plot_tSNE(true_samples, generated_samples, ["True fBm Samples", "Generated fBm Samples"])

    # Chi-2 test for joint distribution of the fractional Brownian NOISE
    c2 = chiSquared_test(T=T, H=H, samples=fBm_to_fBn(generated_samples))
    print("Chi-Squared test: Statistic {} :: Critical Value {}".format(c2[0], c2[1]))


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: DenoisingDiffusion,
                   rng: np.random.Generator, dataSamples: np.ndarray) -> None:
    assert (0. < hurst <= 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0)
    # fBm_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBm_samples, T=timeDim, H=hurst, rng=rng, diffusion=diffusion)


if __name__ == "__main__":
    h, td = 0.7, 2
    numLangSteps = 100
    numDiffSteps = 10
    numSamples = 1000000
    rng = np.random.default_rng()
    try:
        data = np.load("../../data/a_million_fBn_samples_H07_T{}.npy".format(td))
        data = np.cumsum(data, axis=1)[:numSamples // 8, :]
        print(data.shape)
        try:
            file = open("models/trained_fBm_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
                td, numDiffSteps, numLangSteps), 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scoreModel = TimeSeriesNoiseMatching()  # ScoreBasedNoiseMatching(td=td)
            diffusion = ScoreBasedDiffusion(device=torchDevice, model=scoreModel, numLangevinSteps=numLangSteps,
                                            numDiffSteps=numDiffSteps,
                                            rng=rng)
            model = save_and_train_diffusion_model(data,
                                                   model_filename="models/trained_fBm_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
                                                       td, numDiffSteps, numLangSteps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        """ Instantiate training model and generative diffusion """
        torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scoreModel = ScoreBasedNoiseMatching(td=td)
        diffusion = ScoreBasedDiffusion(device=torchDevice, model=scoreModel, numLangevinSteps=numLangSteps,
                                        numDiffSteps=numDiffSteps,
                                        rng=rng)
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save("../../data/a_million_fBn_samples_H07_T{}.npy".format(td), data)
        data = np.cumsum(data, axis=1)[:numSamples // 8, :]
        model = save_and_train_diffusion_model(data,
                                               model_filename="models/trained_fBm_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
                                                   td, numDiffSteps, numLangSteps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=3000, dataSamples=data)
