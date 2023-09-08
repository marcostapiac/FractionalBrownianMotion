import pickle

import numpy as np

from src.generative_modelling.models import ClassOUDiffusion
from src.generative_modelling.models.ClassOUDiffusion import OUDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance
from utils.math_functions import generate_fBn, generate_fBm

LR = 1e-3
NUM_EPOCHS = 400
BATCH_SIZE = 256


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassOUDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, data=data, timeDim=timeDim, timeLim=0,
                                            sampleEps=sampleEps)
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_fBm_performance(true_samples, fBm_samples, h=hurst, td=timeDim, rng=rng, unitInterval=True,
                             evalMarginals=True, annot=True)


if __name__ == "__main__":
    h, td = 0.7, 2
    N = 1000
    Tdiff = 1.
    numSamples = 1200000
    trainEps = 1e-3
    sampleEps = 1e-3
    rng = np.random.default_rng()
    availableData = numSamples
    try:
        data = np.load(config.ROOT_DIR + "data/{}_fBn_samples_H{}_T{}.npy".format(numSamples, h, td))[:availableData, :]
        data = np.cumsum(data, axis=1)
        try:
            file = open(
                config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBm_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}".format(
                    td,
                    N, int(Tdiff), trainEps),
                'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            # scoreModel = NaiveMLP(output_shape=td, enc_shapes=[16, 32], temb_dim=16, dec_shapes=[32, 16], max_diff_steps=N)
            scoreModel = TimeSeriesScoreMatching(diff_embed_size=32, diff_hidden_size=16, max_diff_steps=N)
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, Tdiff=Tdiff)
            model = save_and_train_diffusion_model(data,
                                                   model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBm_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}".format(
                                                       td,
                                                       N, int(Tdiff), trainEps),
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/{}_fBn_samples_H{}_T{}.npy".format(numSamples, h, td), data)
        data = np.cumsum(data, axis=1)[:availableData, :]
        scoreModel = NaiveMLP(output_shape=td, enc_shapes=[32, 32], temb_dim=16, dec_shapes=[32, 32], max_diff_steps=N)
        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, Tdiff=Tdiff)
        model = save_and_train_diffusion_model(data,
                                               model_filename=config.ROOT_DIR + "src/generative_modelling/trained_models/trained_fBm_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}".format(
                                                   td,
                                                   N, int(Tdiff), trainEps),
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 100000
    data = data[:s, :]
    run_experiment(diffusion=model, hurst=h, timeDim=td, rng=rng, dataSize=s, data=data, sampleEps=sampleEps)
