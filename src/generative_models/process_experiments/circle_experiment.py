import pickle

import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassNaiveMLP import NaiveMLP
from src.classes.ClassOUDiffusion import OUDiffusion
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import generate_circles

plt.rcParams.update(bundles.neurips2022())

LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 256


def evaluate_performance(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """ Computes metrics to quantify how close the generated samples are from the desired distribution """
    print(np.mean(true_samples, axis=0))
    print(np.mean(generated_samples, axis=0))
    print([0., 0.])
    print(np.cov(true_samples.T))
    print(np.cov(generated_samples.T))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6)
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Final backward particles"
    ax.set_title(strtitle)
    plt.show()


def run_experiment(timeDim: int, dataSize: int, diffusion: DenoisingDiffusion,
                   rng: np.random.Generator, dataSamples: np.ndarray) -> None:
    """ Perform ancestral sampling given model """
    circle_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0)
    true_samples = generate_circles(T=timeDim, S=dataSize)
    evaluate_performance(true_samples, circle_samples)


if __name__ == "__main__":
    td = 2
    N = 1000
    numSamples = 1000000
    rng = np.random.default_rng()
    try:
        data = np.load("../../data/a_million_circle_samples_T{}.npy".format(td))[:numSamples // 100, :]
        try:
            file = open("models/trained_circle_diffusion_model_T{}_Ndiff{}".format(td, N), 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = NaiveMLP(output_shape=td,
                                  enc_shapes=[32, 32],
                                  t_dim=16,
                                  dec_shapes=[32, 32])
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)
            model = save_and_train_diffusion_model(data,
                                                   model_filename="models/trained_circle_diffusion_model_T{}_Ndiff{}".format(
                                                       td, N),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_circles(T=td, S=numSamples)
        np.save("../../data/a_million_circle_samples_T{}.npy".format(td), data)
        data = data[:numSamples // 100, :]
        scoreModel = NaiveMLP(output_shape=td,
                              enc_shapes=[32, 32],
                              t_dim=16,
                              dec_shapes=[32, 32])
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)
        model = save_and_train_diffusion_model(data,
                                               model_filename="models/trained_circle_diffusion_model_T{}_Ndiff{}".format(
                                                   td, N),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    run_experiment(diffusion=model, timeDim=td, rng=rng, dataSize=3000, dataSamples=data)
