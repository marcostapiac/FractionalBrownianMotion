import pickle

import numpy as np

from src.generative_modelling.models import ClassVPSDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import \
    TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance
from utils.math_functions import generate_fBn, generate_fBm

LR = 1e-3
NUM_EPOCHS = 20
BATCH_SIZE = 64


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: ClassVPSDEDiffusion,
                   rng: np.random.Generator, sampleEps: float, data: np.ndarray) -> None:
    assert (0. < hurst < 1.)
    """ Perform ancestral sampling given model """
    fBm_samples = diffusion.reverse_process(dataSize=dataSize, timeDim=timeDim, timeLim=0,
                                            sampleEps=sampleEps, data=data, sigNoiseRatio=0.075, numLangevinSteps=0)
    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    annot = False if timeDim > 16 else True
    evalMargs = False if timeDim > 16 else True
    evaluate_fBm_performance(true_samples, fBm_samples, h=hurst, td=timeDim, rng=rng, unitInterval=True, annot=annot,
                             evalMarginals=evalMargs)


if __name__ == "__main__":
    # Data parameters
    h, td = 0.7, 32
    numSamples = 1000000
    availableData = 124913 * 10

    # Training parameters
    trainEps = 1e-3
    sampleEps = 1e-3
    N = 1000 * max(1, int(np.log2(td) - 1))  # if td > 2
    Tdiff = 1.
    beta_min = 0.1
    beta_max = 20.

    # MLP Architecture parameters
    temb_dim = 32
    enc_shapes = [8, 16, 32]
    dec_shapes = enc_shapes[::-1]
    # TSM Architecture parameters ___ or 124913
    residual_layers = 10
    residual_channels = 8
    diff_hidden_size = 32

    mlpFileName = config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_fBm_VPSDE_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_BetaMax{:.4f}_BetaMin{:.4f}_TembDim{}_EncShapes{}".format(
        td,
        N, Tdiff, trainEps, beta_max, beta_min, temb_dim, enc_shapes)
    tsmFileName = config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_fBm_VPSDE_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_BetaMax{:.4f}_BetaMin{:.4f}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(
        td,
        N, Tdiff, trainEps, beta_max, beta_min, temb_dim, residual_layers, residual_channels, diff_hidden_size)
    modelChoice = "TSM"  # "TSM"
    modelFileName = mlpFileName if modelChoice == "MLP" else tsmFileName

    rng = np.random.default_rng()
    scoreModel = TimeSeriesNoiseMatching(diff_embed_size=temb_dim, diff_hidden_size=diff_hidden_size, max_diff_steps=N,
                                         residual_layers=residual_layers,
                                         residual_channels=residual_channels) \
        if modelChoice == "TSM" else NaiveMLP(temb_dim=temb_dim, max_diff_steps=N, output_shape=td,
                                              enc_shapes=enc_shapes,
                                              dec_shapes=dec_shapes)
    diffusion = VPSDEDiffusion(device="cpu", model=scoreModel, numDiffSteps=N, endDiffTime=Tdiff, rng=rng,
                               trainEps=trainEps, betaMax=beta_max, betaMin=beta_min)
    try:
        data = np.load(config.ROOT_DIR + "data/{}_fBn_samples_H{}_T{}.npy".format(numSamples, h, td))[:availableData, :]
        data = np.cumsum(data, axis=1)
        print(np.cov(data, rowvar=False))
        try:
            file = open(modelFileName, 'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            model = save_and_train_diffusion_model(data,
                                                   model_filename=modelFileName,
                                                   batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR,
                                                   diffusion=diffusion)
    except FileNotFoundError:
        data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
        np.save(config.ROOT_DIR + "data/{}_fBn_samples_H{}_T{}.npy".format(numSamples, h, td), data)
        data = data[:availableData, :].cumsum(axis=1)
        print(np.cov(data, rowvar=False))
        model = save_and_train_diffusion_model(data,
                                               model_filename=modelFileName,
                                               batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)
    s = 30000
    data = data[:s, :]
    run_experiment(diffusion=model, timeDim=td, dataSize=s, data=data, sampleEps=sampleEps, hurst=h, rng=rng)
