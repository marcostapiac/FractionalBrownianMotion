import pickle

import numpy as np
import torch

from src.classes import ClassVPSDEDiffusion
from src.classes.ClassVPSDEDiffusion import VPSDEDiffusion
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance, compare_fBm_to_approximate_fBm, compare_fBm_to_normal
from utils.math_functions import generate_fBn, generate_fBm

LR = 1e-3
NUM_EPOCHS = 1
BATCH_SIZE = 256



def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVPSDEDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps), data=data,
                                            sigNoiseRatio=0.00075, numLangevinSteps=10)
    evaluate_fBm_performance(true_samples, fBm_samples, h=hurst, td=timeDim, rng=rng, unitInterval=True, annot=True, evalMarginals=False)
    compare_fBm_to_approximate_fBm(fBm_samples, h=hurst, td=timeDim, rng=rng)
    if timeDim >= 8: compare_fBm_to_normal(fBm_samples, td=timeDim, rng=rng)

if __name__ == "__main__":
    h, td = 0.7, 8
    N = 1000
    numSamples = 600000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/six_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td))[:numSamples//1, :]
        data = np.cumsum(data, axis=1)
        print(np.cov(data, rowvar=False))
        try:
            file = open(
                config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
            scoreModel = TimeSeriesNoiseMatching()  # input_dim= BATCH_SIZE if len(data.shape) == 2 else data.shape[-1])
            diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps, noiseFactor=1.)
            print("Hi2")
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        print("Hi")
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/six_hundred_thousand_fBn_samples_H{}_T{}.npy".format(h, td), data)
        data = data[:numSamples // 1, :].cumsum(axis=1)
        scoreModel = TimeSeriesNoiseMatching() #input_dim= BATCH_SIZE if len(data.shape) == 2 else data.shape[-1])
        diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps, noiseFactor=1.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_models/models/trained_fBm_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 20000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
