import pickle

import numpy as np
import torch

from src.generative_modelling.models import ClassVPSDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import \
    TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_SDE_performance
from utils.math_functions import generate_CEV

LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 256

h, td = 0.7, 2
muU = 1.
muX = 2.
alpha = 1.
sigmaX = 0.5
X0 = 1.
U0 = 0.


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVPSDEDiffusion, sampleEps: float,
                   data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    cev_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0, data=data[:dataSize],
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps),
                                            sigNoiseRatio=0.04, numLangevinSteps=10)
    true_samples = generate_CEV(H=hurst, T=timeDim, S=100000, alpha=alpha, muX=muX, X0=X0, sigmaX=sigmaX, rng=rng,
                                U0=U0, muU=muU)
    evaluate_SDE_performance(true_samples=true_samples, generated_samples=cev_samples, td=timeDim)


if __name__ == "__main__":
    N = 1000
    numSamples = 200000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        data = np.load(
            config.ROOT_DIR + "data/two_hundred_thousand_CEV_samples_H{}_T{}_alpha{}_sigmaX{}_muX{}.npy".format(h, td,
                                                                                                                alpha,
                                                                                                                sigmaX,
                                                                                                                muX))
        data = data[:numSamples // 1, :]
        print(np.cov(data, rowvar=False))
        try:
            file = open(
                config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
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
                                                   model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_CEV(H=h, T=td, S=numSamples, rng=rng, X0=X0, U0=U0, alpha=alpha, sigmaX=sigmaX, muU=muU,
                            muX=muX)
        np.save(
            config.ROOT_DIR + "data/two_hundred_thousand_CEV_samples_H{}_T{}_alpha{}_sigmaX{}_muX{}.npy".format(h, td,
                                                                                                                alpha,
                                                                                                                sigmaX,
                                                                                                                muX),
            data)
        data = data[:numSamples // 1, :]
        # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        scoreModel = TimeSeriesNoiseMatching()
        diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                                   noiseFactor=1.)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_VPSDE_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    run_experiment(diffusion=model, hurst=h, timeDim=td, data=data, dataSize=s, sampleEps=sampleEps)
