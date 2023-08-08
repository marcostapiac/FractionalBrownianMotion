import pickle

import numpy as np
import torch

from src.classes import ClassVESDEDiffusion
from src.classes.ClassVESDEDiffusion import VESDEDiffusion
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance
from utils.math_functions import generate_fBn, generate_fBm

LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 256


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVESDEDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0, data=data,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps),
                                            sigNoiseRatio=0.01, numLangevinSteps=10)  # 0.007 for good Chi2 test, 0.01 for good stats
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_fBm_performance(true_samples, fBm_samples, h=hurst, td=timeDim, rng=rng, unitInterval=True)


if __name__ == "__main__":
    h, td = 0.7, 2
    numSamples = 200000
    trainEps = 1e-3
    sampleEps = 1e-3
    N = 100
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h,td))[:numSamples // 1, :]
        data = data.cumsum(axis=1)
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_fBm_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
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
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(T=td, S=numSamples, H=h, rng=rng)
        np.save(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h,td), data)
        data = data[:numSamples // 1, :].cumsum(axis=1)
        # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                   noiseFactor=2.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_SLGDVESDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
