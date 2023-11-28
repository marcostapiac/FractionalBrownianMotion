import torch
from ml_collections import ConfigDict
from tqdm import tqdm
import numpy as np
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
import matplotlib.pyplot as plt
from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.math_functions import optimise_whittle, reduce_to_fBn
from utils.plotting_functions import plot_histogram

def run_early_stopping(config:ConfigDict)->None:

    rng = np.random.default_rng()
    H = config.hurst
    T = config.timeDim

    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    fbn = FractionalBrownianNoise(H=H, rng=rng)

    exact_samples = np.zeros((config.dataSize, T))
    for j in tqdm(range(config.dataSize), desc="Exact Sampling ::", dynamic_ncols=False, position=0):
        exact_samples[j, :] = fbn.circulant_simulation(N_samples=T).cumsum()

    synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                     scoreModel=scoreModel,
                                     config=config).cpu().numpy().reshape((config.dataSize, T))
    true_Hs = []
    synth_Hs = []
    approx_true_fBn = reduce_to_fBn(exact_samples, reduce=True)
    approx_fBn = reduce_to_fBn(synth_samples, reduce=True)
    for j in tqdm(range(config.dataSize),desc="Estimating Hurst Parameters ::", dynamic_ncols=False, position=0):
        ht = optimise_whittle(data=approx_true_fBn, idx=j)
        h = optimise_whittle(data=approx_fBn, idx=j)
        true_Hs.append(ht)
        synth_Hs.append(h)

    print("Exact:", np.mean(true_Hs), np.std(true_Hs))
    print("Synth:", np.mean(synth_Hs), np.std(synth_Hs))

    fig, ax = plt.subplots()
    plot_histogram(np.array(true_Hs), num_bins=100, xlabel="H", ylabel="density",
                   plottitle="Histogram of exact samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.savefig("myplot1.png")
    plt.show()

    fig, ax = plt.subplots()
    plot_histogram(np.array(synth_Hs), num_bins=100, xlabel="H", ylabel="density",
                   plottitle="Histogram of synthetic samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.savefig("myplot2.png")
    plt.show()

if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config
    config = get_config()
    assert(0<config.hurst<1)
    assert(config.early_stop_idx == 1)
    config.dataSize = 10000
    run_early_stopping(config)