
import pickle

import numpy as np

from src.classes.ClassOUDiffusion import OUDiffusion
from src.classes.TimeDependentScoreNetworks.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils import config
from utils.data_processing import save_and_train_diffusion_model, evaluate_fBm_performance, evaluate_circle_performance
from utils.math_functions import generate_fBm, generate_circles

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 256

if __name__ == "__main__":
    td =  2
    numSamples = 3000000

    # Training parameters
    trainEps = 1e-3
    sampleEps = 1e-3
    cnoise = 0.03

    # Diffusion parameters
    N = 1000
    Tdiffs = [0.6]#, 1., 3.5, 10.]
    rng = np.random.default_rng()

    # TSM Architecture parameters
    temb_dim = 32
    residual_layers = 10
    residual_channels = 8
    diff_hidden_size = 32
    scoreModel = TimeSeriesNoiseMatching(diff_embed_size=temb_dim, diff_hidden_size=diff_hidden_size, max_diff_steps=N,
                                         residual_layers=residual_layers,
                                         residual_channels=residual_channels)

    availableData = 11313 * 10

    for Tdiff in Tdiffs:
        modelFileName = config.ROOT_DIR + "src/generative_models/models/trained_TSM_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(td,N, Tdiff, trainEps, temb_dim, residual_layers, residual_channels, diff_hidden_size)

        diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, Tdiff=Tdiff)
        data = np.load(config.ROOT_DIR + "data/{}_noisy_circle_samples.npy".format(numSamples))[:availableData, :]
        try:
            file = open(modelFileName,'rb')
            model = pickle.load(file)
        except FileNotFoundError:
            diffusion = OUDiffusion(device="cpu", model=scoreModel, N=N, rng=rng, trainEps=trainEps, Tdiff=Tdiff)
            model = save_and_train_diffusion_model(data,model_filename=modelFileName, batch_size=BATCH_SIZE, nEpochs=NUM_EPOCHS, lr=LR, diffusion=diffusion)

        # Given trained models, compute KL divergence between final samples
        circle_samples = model.reverse_process(dataSize=int(1e4), data=data, timeDim=td, timeLim=0,sampleEps=sampleEps)
        true_samples = generate_circles(T=td, S=int(1e4), noise=cnoise)
        evaluate_circle_performance(true_samples, circle_samples,  td=td)

