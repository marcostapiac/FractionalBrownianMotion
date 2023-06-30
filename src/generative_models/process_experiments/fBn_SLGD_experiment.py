import pickle

import numpy as np
import torch
from scipy.stats import chi2
from tqdm import tqdm

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassSLGDDiffusion import ScoreBasedDiffusion
from src.classes.ClassScoreBasedNoiseMatching import ScoreBasedNoiseMatching
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils.data_processing import save_and_train_diffusion_model
from utils.plotting_functions import plot_diffusion_marginals

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 256


def generate_fBn(H: float, T: int, S: int, rng: np.random.Generator) -> np.array:
    generator = FractionalBrownianNoise(H=H, rng=rng)
    data = np.empty(shape=(S, T))
    for i in tqdm(range(S)):
        data[i, :] = generator.circulant_simulation(N_samples=T)
    return data


def chiSquared_test(T: int, H: float, samples: np.ndarray, invL: np.ndarray = None) -> [float, float, float]:
    def standardise_sample(fBn_sample, invL):
        return np.squeeze(invL @ fBn_sample)

    def chiSquared(fBn_sample, invL):
        standard_sample = standardise_sample(fBn_sample, invL)
        return np.sum([i ** 2 for i in standard_sample])

    fbn = FractionalBrownianNoise(H)
    invL = invL if (invL is not None) else np.linalg.inv(
        np.linalg.cholesky(np.atleast_2d(
            [[fbn.covariance(i - j) for j in range(T)] for i in tqdm(range(T))])))
    alpha = 0.05
    S = samples.shape[0]
    critUpp = chi2.ppf(q=1. - 0.5 * alpha, df=S * T - 1)  # Upper alpha quantile, and dOf = T - 1
    critLow = chi2.ppf(q=0.5 * alpha, df=S * T - 1)  # Lower alpha quantile, and d0f = T -1
    ts = []
    for i in tqdm(range(S)):
        tss = chiSquared(samples[i, :], invL)
        ts.append(tss)
    return critLow, np.sum(ts), critUpp


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, T: int, H: float,
                         rng: np.random.Generator, diffusion: DenoisingDiffusion) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """
    print(np.mean(true_samples, axis=0))
    print(np.mean(generated_samples, axis=0))
    print([0., 0.])
    print(np.cov(true_samples.T))
    print(np.cov(generated_samples.T))
    print([[1., 0.5 * 2 ** 1.4 - 1.], [0.5 * (2 ** 1.4) - 1., 1.]])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=T, H=H, samples=generated_samples)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    # Marginal plots
    plot_diffusion_marginals(true_samples, generated_samples, timeDim=T, diffTime=0)

    # Visualise data
    # plot_tSNE(true_samples, generated_samples, ["True fBn Samples", "Generated fBn Samples"])


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: DenoisingDiffusion,
                   rng: np.random.Generator, dataSamples: np.ndarray) -> None:
    assert (0. < hurst <= 1.)
    fBn_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0)
    true_samples = generate_fBn(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBn_samples, T=timeDim, H=hurst, rng=rng, diffusion=diffusion)


if __name__ == "__main__":
    h, td = 0.7, 2
    numLangSteps = 100
    numDiffSteps = 10
    numSamples = 1000000
    rng = np.random.default_rng()
    try:
        data = np.load("../../data/a_million_fBn_samples_H07_T{}.npy".format(td))[:numSamples // 8, :]
        try:
            file = open("models/trained_fBn_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
                td, numDiffSteps, numLangSteps), 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            torchDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            scoreModel = TimeSeriesNoiseMatching()  # ScoreBasedNoiseMatching(td=td)
            diffusion = ScoreBasedDiffusion(device=torchDevice, model=scoreModel, numLangevinSteps=numLangSteps,
                                            numDiffSteps=numDiffSteps,
                                            rng=rng)
            model = save_and_train_diffusion_model(data,
                                                   model_filename="models/trained_fBn_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
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
        data = data[:numSamples // 8, :]
        model = save_and_train_diffusion_model(data,
                                               model_filename="models/trained_fBn_score_diffusion_model_T{}_Ndiff{}_LangSteps{}".format(
                                                   td, numDiffSteps, numLangSteps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=3000, dataSamples=data)
