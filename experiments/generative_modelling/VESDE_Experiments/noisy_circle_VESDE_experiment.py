import pickle

import numpy as np

from src.generative_modelling.models import ClassVESDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import \
    TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_circle_performance
from utils.math_functions import generate_circles

LR = 1e-3
NUM_EPOCHS = 400
BATCH_SIZE = 256


def run_experiment(timeDim: int, dataSize: int, diffusion: ClassVESDEDiffusion, sampleEps: float, circlenoise: float,
                   data: np.ndarray) -> None:
    """ Perform ancestral sampling given model """
    circle_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0, numLangevinSteps=0,
                                               sampleEps=sampleEps, data=data, sigNoiseRatio=0.)
    true_samples = generate_circles(T=timeDim, S=dataSize, noise=circlenoise)
    evaluate_circle_performance(true_samples, circle_samples, td=timeDim)


if __name__ == "__main__":
    # Data parameters
    td = 2
    numSamples = 3000000
    availableData = 11313 * 10
    cnoise = 0.03

    # Training data
    trainEps = 1e-5
    sampleEps = 1e-5
    N = 1000
    Tdiff = 1.

    # Model parameters
    std_max = 12.
    std_min = 0.01

    # MLP Architecture parameters (13704)
    temb_dim = 32
    enc_shapes = [8, 16, 32]
    dec_shapes = enc_shapes[::-1]
    # TSM Architecture parameters (11313)
    residual_layers = 10
    residual_channels = 8
    diff_hidden_size = 32

    mlpFileName = config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_noisy_circle_VESDE_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_StdMax{:.4f}_StdMin{:.4f}_TembDim{}_EncShapes{}".format(
        td,
        N, Tdiff, trainEps, std_max, std_min, temb_dim, enc_shapes)
    tsmFileName = config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_noisy_circle_VESDE_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_StdMax{:.4f}_StdMin{:.4f}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(
        td,
        N, Tdiff, trainEps, std_max, std_min, temb_dim, residual_layers, residual_channels, diff_hidden_size)
    modelChoice = "TSM"  # "TSM"
    modelFileName = mlpFileName if modelChoice == "MLP" else tsmFileName
    rng = np.random.default_rng()

    scoreModel = TimeSeriesNoiseMatching(diff_embed_size=temb_dim, diff_hidden_size=diff_hidden_size, max_diff_steps=N,
                                         residual_layers=residual_layers,
                                         residual_channels=residual_channels) \
        if modelChoice == "TSM" else NaiveMLP(temb_dim=temb_dim, max_diff_steps=N, output_shape=td,
                                              enc_shapes=enc_shapes,
                                              dec_shapes=dec_shapes)
    diffusion = VESDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, rng=rng, trainEps=trainEps,
                               stdMax=std_max, stdMin=std_min, endDiffTime=Tdiff)
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
