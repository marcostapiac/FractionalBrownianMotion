import pickle

import numpy as np
import torch

from src.generative_modelling.models import ClassOUDiffusion
from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils import project_config
from utils.data_processing import save_and_train_diffusion_model, evaluate_SDE_performance
from utils.math_functions import generate_CEV

LR = 1e-4
NUM_EPOCHS = 400
BATCH_SIZE = 256

h, td = 0.7, 2
muU = 1.
muX = 2.
alpha = 1.
sigmaX = 0.5
X0 = 1.
U0 = 0.


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassOUDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    cev_samples = diffusion.reverse_process(dataSize=dataSize, data=data, timeDim=timeDim, timeLim=0,
                                            reverseTimes=torch.linspace(start=1., end=sampleEps,
                                                                        steps=diffusion.numDiffSteps))
    true_samples = generate_CEV(H=hurst, T=timeDim, S=dataSize, rng=rng, X0=X0, U0=U0, alpha=alpha, sigmaX=sigmaX,
                                muU=muU, muX=muX)
    evaluate_SDE_performance(true_samples, cev_samples, td=timeDim)


if __name__ == "__main__":
    N = 1000
    numSamples = 1000000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    try:
        data = np.load(config.ROOT_DIR + "data/a_million_CEV_samples_H07_T{}.npy".format(td))[:numSamples // 50, :]
        try:
            file = open(
                config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                    td,
                    N, trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            scoreModel = TimeSeriesScoreMatching()
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_CEV(H=h, T=td, S=numSamples, rng=rng, X0=X0, U0=U0, alpha=alpha, sigmaX=sigmaX, muU=muU,
                            muX=muX)
        np.save(config.ROOT_DIR + "data/a_million_CEV_samples_H07_T{}.npy".format(td), data)
        # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=32, dec_shapes=[32, 32])
        data = data[:numSamples // 50, :]
        scoreModel = TimeSeriesScoreMatching()
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_CEV_OU_model_T{}_Ndiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=s, data=data, sampleEps=sampleEps)
