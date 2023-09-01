import pickle

import numpy as np
import torch

from src.classes import ClassOUDiffusion
from src.classes.ClassOUDiffusion import OUDiffusion
from src.classes.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_circle_performance
from utils.math_functions import generate_circles

LR = 1e-3
NUM_EPOCHS = 400
BATCH_SIZE = 256


def run_experiment(timeDim: int, dataSize: int, diffusion: ClassOUDiffusion, sampleEps: float,
                   circlenoise:float,data: np.ndarray) -> None:
    """ Perform ancestral sampling given model """
    circle_samples = diffusion.reverse_process(dataSize=dataSize, data=data, timeDim=timeDim, timeLim=0, sampleEps=sampleEps)
    true_samples = generate_circles(T=timeDim, S=dataSize, noise=circlenoise)
    evaluate_circle_performance(true_samples, circle_samples, td=timeDim)


if __name__ == "__main__":
    # Data parameters
    td = 2
    numSamples = 3000000
    availableData = 13704 * 10
    cnoise = 0.03

    # Training parameters
    trainEps = 1e-3
    sampleEps = 1e-3

    # Diffusion parameters
    N = 1000
    Tdiff = 1.
    rng = np.random.default_rng()

    # MLP Architecture parameters
    temb_dim = 32
    enc_shapes = [8, 16, 32]
    dec_shapes = enc_shapes[::-1]

    # TSM Architecture parameters
    residual_layers = 10
    residual_channels = 8
    diff_hidden_size = 32

    mlpFileName = config.ROOT_DIR + "src/generative_models/models/trained_MLP_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_TembDim{}_EncShapes{}".format(
        td,
        N, Tdiff, trainEps ,temb_dim, enc_shapes)
    tsmFileName = config.ROOT_DIR + "src/generative_models/models/trained_TSM_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(
        td,
        N, Tdiff, trainEps, temb_dim, residual_layers, residual_channels, diff_hidden_size)

    modelChoice = "MLP"  # "TSM"
    modelFileName = mlpFileName if modelChoice == "MLP" else tsmFileName
    scoreModel = TimeSeriesNoiseMatching(diff_embed_size=temb_dim, diff_hidden_size=diff_hidden_size, max_diff_steps=N,
                                         residual_layers=residual_layers,
                                         residual_channels=residual_channels) \
        if modelChoice == "TSM" else NaiveMLP(temb_dim=temb_dim, max_diff_steps=N, output_shape=td,
                                              enc_shapes=enc_shapes,
                                              dec_shapes=dec_shapes)
    diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, Tdiff=Tdiff)

    try:
        data = np.load(config.ROOT_DIR + "data/{}_noisy_circle_samples.npy".format(numSamples))[:availableData, :]
        try:
            file = open(modelFileName, 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            model = save_and_train_diffusion_model(data,
                                                   model_filename=modelFileName,
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_circles(T=td, S=numSamples, noise=cnoise)
        np.save(config.ROOT_DIR + "data/{}_noisy_circle_samples.npy".format(numSamples), data)
        data = data[:availableData, :]
        model = save_and_train_diffusion_model(data,
                                               model_filename=modelFileName,
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, circlenoise=cnoise)
