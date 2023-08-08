import pickle

import numpy as np
import torch

from src.classes import ClassOUDiffusion
from src.classes.ClassOUDiffusion import OUDiffusion
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance
from utils.math_functions import generate_fBn, generate_fBm

LR = 1e-3
NUM_EPOCHS = 10
BATCH_SIZE = 256


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassOUDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, data=data, timeDim=timeDim, timeLim=0,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps))
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_fBm_performance(true_samples, fBm_samples, h=hurst, td=timeDim, rng=rng)


if __name__ == "__main__":
    h, td = 0.7, 2
    N = 500
    numSamples = 1000000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/a_million_fBn_samples_H07_T{}.npy".format(td))[:numSamples // 5, :]
        data = np.cumsum(data, axis=1)
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_fBm_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = TimeSeriesNoiseMatching()
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/a_million_fBn_samples_H07_T{}.npy".format(td), data)
        data = np.cumsum(data, axis=1)[:numSamples // 5, :]
        # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 1000
    data = data[:s, :]
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=s, data=data, sampleEps=sampleEps)
