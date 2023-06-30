import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import kstest
from tqdm import tqdm

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils.data_processing import save_and_train_diffusion_model
from utils.plotting_functions import qqplot, plot_tSNE

LR = 1e-4
NUM_EPOCHS = 20
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


def check_convergence_at_diffTime(diffusion: DenoisingDiffusion, t: int, dataSamples: np.ndarray) -> [np.ndarray,
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
    return forward_samples_at_t, backward_samples_at_t, labels


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray, T: int, H: float,
                         rng: np.random.Generator, diffusion: DenoisingDiffusion) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """
    print([np.mean(true_samples[:, 0].flatten()), np.mean(true_samples[:, 1].flatten())])
    print([np.mean(generated_samples[:, 0].flatten()), np.mean(generated_samples[:, 1].flatten())])
    print([0., 0.])
    print(np.var(true_samples, axis=0))
    print(np.mean((true_samples[:, 0].flatten() - np.mean(true_samples[:, 0].flatten())) * (
            true_samples[:, 1].flatten() - np.mean(true_samples[:, 1].flatten()))))
    print(np.var(generated_samples, axis=0))
    print(np.mean((generated_samples[:, 0].flatten() - np.mean(generated_samples[:, 0].flatten())) * (
            generated_samples[:, 1].flatten() - np.mean(generated_samples[:, 1].flatten()))))
    print([[1., 0.5 * 2 ** 1.4 - 1.], [0.5 * (2 ** 1.4) - 1., 1.]])

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

    # Check convergence to prior
    true_norm_rvs, prior_samples, _ = check_convergence_at_diffTime(diffusion, t=diffusion.numDiffSteps - 1,
                                                                    dataSamples=true_samples)
    plot_tSNE(true_norm_rvs, prior_samples, ["True Gaussian Samples", "Generated Gaussian Samples"])

    # Check convergence at other diffusion times
    N = diffusion.numDiffSteps
    forward_samples, backward_samples, labels = check_convergence_at_diffTime(diffusion, t=N - N // 10,
                                                                              dataSamples=true_samples)
    plot_tSNE(forward_samples, backward_samples, labels)

    forward_samples, backward_samples, labels = check_convergence_at_diffTime(diffusion, t=N - N // 2,
                                                                              dataSamples=true_samples)
    plot_tSNE(forward_samples, backward_samples, labels)

    # Permutation test for kernel statistic
    # print("MMD Permutation test: p-value {}".format(
    #    permutation_test(true_samples, generated_samples, compute_statistic=MMD_statistic, num_permutations=1000)))
    # Permutation test for energy statistic
    # print("Energy Permutation test: p-value {}".format(
    #    permutation_test((true_samples-np.mean(true_samples, axis=0))/np.std(true_samples, axis=0), (generated_samples-np.mean(generated_samples, axis=0))/np.std(generated_samples, axis=0), compute_statistic=energy_statistic, num_permutations=1000)))


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: DenoisingDiffusion,
                   rng: np.random.Generator, dataSamples: np.ndarray) -> None:
    assert (0. < hurst <= 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0)
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBm_samples, T=timeDim, H=hurst, rng=rng, diffusion=diffusion)


if __name__ == "__main__":
    h, td = 0.7, 2
    N = 50
    numSamples = 1000000
    rng = np.random.default_rng()
    try:
        data = np.load("../../data/a_million_fBn_samples_H07_T{}.npy".format(td))
        data = np.cumsum(data, axis=1)[:numSamples // 50, :]
        print(data.shape)
        try:
            file = open("models/trained_fBm_diffusion_model_T{}_Ndiff{}".format(td, N), 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = TimeSeriesNoiseMatching()
            diffusion = DenoisingDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)
            model = save_and_train_diffusion_model(data,
                                                   model_filename="models/trained_fBm_diffusion_model_T{}_Ndiff{}".format(
                                                       td, N),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save("../../data/a_million_fBn_samples_H07_T{}.npy".format(td), data)
        data = np.cumsum(data, axis=1)[:numSamples // 50, :]
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = DenoisingDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)
        model = save_and_train_diffusion_model(data,
                                               model_filename="models/trained_fBm_diffusion_model_T{}_Ndiff{}".format(
                                                   td, N),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=10000, dataSamples=data)
