import matplotlib
import numpy as np
import torch
from scipy.signal import periodogram as psd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from sklearn.linear_model import LinearRegression

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import run_fBm_score
from utils.math_functions import fBn_spectral_density, optimise_whittle, reduce_to_fBn
from utils.plotting_functions import plot_histogram

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})


def estimate_and_plot_H(sample, N, H):
    ks = np.linspace(1, N, num=N, dtype=int)
    freqs = np.pi * ks / N
    Is = (np.abs(np.fft.fft(sample)) ** 2) / N  # fBn_spectral_density(hurst=H, N=2 * N + 1)
    fig, ax = plt.subplots()
    ax.scatter(np.log10(freqs), np.log10(Is), s=2, color="blue", label="Periodogram")
    fit = np.polyfit(np.log10(freqs)[:int(0.05 * N)], np.log10(Is)[:int(0.05 * N)], 1)
    print("Estimated H from exact sample:", 0.5 * (1. - fit[0]))
    ax.plot(np.unique(np.log10(freqs)),
            np.poly1d(fit)(np.unique(np.log10(freqs))),
            label="Straight Line Fit")
    plt.legend()
    plt.show()

from configs.VESDE.fBm_T32_H07 import get_config

config = get_config()
rng = np.random.default_rng()
H = config.hurst
T = config.timeDim

diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
    *config.model_parameters)
scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
true_Hs = []
fbn = FractionalBrownianNoise(H=H, rng=rng)
config.dataSize = 5000
synth_samples = np.zeros((config.dataSize, T))
exact_samples = np.zeros((config.dataSize, T))
for j in tqdm(range(config.dataSize)):
    exact_samples[j,:] = fbn.circulant_simulation(N_samples=T)

synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion, scoreModel=scoreModel,
                          config=config).numpy().reshape((config.dataSize,T))
true_Hs = []
synth_Hs = []
synth_Hs_no_mean = []
means = np.mean(synth_samples, axis=0)
assert (means.shape[0] == T)
print("Synth fBm mean:", means)
print("Exact fBn mean:", np.mean(exact_samples, axis=0))
for j in tqdm(range(config.dataSize)):
    ht = optimise_whittle(data=exact_samples, idx=j)
    approx_fBn = reduce_to_fBn(synth_samples, reduce=True)
    approx_fBn_mean = np.mean(approx_fBn, axis=0)
    print("Synth fBn mean:", means)
    assert (approx_fBn_mean.shape[0] == T)
    h = optimise_whittle(data=approx_fBn, idx=j)
    mean_red_approx_fBn = approx_fBn - approx_fBn_mean
    h1 = optimise_whittle(data=mean_red_approx_fBn, idx=j)
    true_Hs.append(ht)
    synth_Hs.append(h)
    synth_Hs_no_mean.append(h1)

print("Exact:", np.mean(true_Hs), np.std(true_Hs))
print("Synth:", np.mean(synth_Hs), np.std(synth_Hs))
print("Synth Mean Removed:",np.mean(synth_Hs_no_mean), np.std(synth_Hs_no_mean))

fig, ax = plt.subplots()
plot_histogram(np.array(true_Hs), num_bins=100, xlabel="H", ylabel="density",
                   plottitle="Histogram of exact samples' estimated Hurst parameter", fig=fig, ax=ax)
plt.show()

fig, ax = plt.subplots()
plot_histogram(np.array(synth_Hs), num_bins=100, xlabel="H", ylabel="density",
                   plottitle="Histogram of synth samples' estimated Hurst parameter", fig=fig, ax=ax)
plt.show()

fig, ax = plt.subplots()
plot_histogram(np.array(synth_Hs_no_mean), num_bins=100, xlabel="H", ylabel="density",
                   plottitle="Histogram of synth samples' mean-reduced estimated Hurst parameter", fig=fig, ax=ax)
plt.show()

