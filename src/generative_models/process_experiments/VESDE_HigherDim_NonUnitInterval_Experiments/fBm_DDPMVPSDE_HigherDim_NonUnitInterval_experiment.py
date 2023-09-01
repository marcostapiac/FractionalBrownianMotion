import pickle

import numpy as np
import torch

from src.classes import ClassVPSDEDiffusion
from src.classes.ClassVPSDEDiffusion import VPSDEDiffusion
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model
from utils.math_functions import generate_fBn, chiSquared_test, generate_fBm, fBm_to_fBn
from utils.plotting_functions import plot_dataset, plot_final_diffusion_marginals

LR = 1e-3
NUM_EPOCHS = 1000
BATCH_SIZE = 256


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVPSDEDiffusion,
                   rng: np.random.Generator, data: np.ndarray, sampleEps: float) -> None:
    assert (0. < hurst <= 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0, data=data,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps),
                                            sigNoiseRatio=0.01,numLangevinSteps=0)
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)*(timeDim**hurst)
    evaluate_fBm_HigherDim_performance(true_samples, fBm_samples,h=hurst,td=timeDim, rng=rng)


if __name__ == "__main__":
    h, td = 0.7, 256
    numSamples = 200000
    trainEps = 1e-3
    sampleEps = 1e-3
    N = 1000
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/two_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h,td))[:numSamples // 1, :]
        data = (td**h)*data
        data = data.cumsum(axis=1)
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_NUI_model_T{}_Ndiff{}_trainEps{:.0e}".format(
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
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_NUI_model_T{}_Ndiff{}_trainEps{:.0e}".format(
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
        diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                   noiseFactor=1.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
