import pickle

import numpy as np
import torch

from src.generative_modelling.models import ClassVPSDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import \
    TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBn_performance
from utils.math_functions import generate_fBn

LR = 1e-3
NUM_EPOCHS = 40
BATCH_SIZE = 256


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVPSDEDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    true_samples = generate_fBn(H=hurst, T=timeDim, S=dataSize, rng=rng)

    fBn_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps), data=data,
                                            sigNoiseRatio=0.01, numLangevinSteps=10)

    evaluate_fBn_performance(true_samples, fBn_samples, h=hurst, td=timeDim, rng=rng)


if __name__ == "__main__":
    h, td = 0.7, 2
    N = 100
    numSamples = 1000000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td))[
               :numSamples // 1, :]
        try:
            file = open(
                config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBn_DDPMVPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
            scoreModel = TimeSeriesNoiseMatching()
            diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                       noiseFactor=1.)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBn_DDPMVPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}.npy".format(h, td), data)
        data = data[:numSamples // 1, :]
        # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, noiseFactor=1.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBn_DDPMVPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
