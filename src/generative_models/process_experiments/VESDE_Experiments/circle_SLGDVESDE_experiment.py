import pickle

import numpy as np
import torch

from src.classes import ClassVESDEDiffusion
from src.classes.ClassVESDEDiffusion import VESDEDiffusion
from src.classes.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_circle_performance
from utils.math_functions import generate_circles

LR = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 256


def run_experiment(timeDim: int, dataSize: int, diffusion: ClassVESDEDiffusion, sampleEps: float,
                   data: np.ndarray) -> None:
    """ Perform ancestral sampling given model """
    circle_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                               reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                           steps=diffusion.numDiffSteps), data=data,
                                               sigNoiseRatio=.1)
    true_samples = generate_circles(T=timeDim, S=dataSize)
    evaluate_circle_performance(true_samples, circle_samples, td=timeDim)


if __name__ == "__main__":
    td = 2
    numSamples = 1000000
    trainEps = 1e-3
    sampleEps = 1e-3
    N = 100
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/a_million_circle_samples_T{}.npy".format(td))[:numSamples // 50, :]
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_circle_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
            scoreModel = TimeSeriesNoiseMatching()
            diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                       noiseFactor=2.)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_circle_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_circles(T=td, S=numSamples)
        np.save(config.ROOT_DIR + "data/a_million_circle_samples_T{}.npy".format(td), data)
        data = data[:numSamples // 50, :]
        scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                   noiseFactor=2.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_circle_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps)
