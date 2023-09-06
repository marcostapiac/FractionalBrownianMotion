# Data parameters
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import project_config
from utils.plotting_functions import plot_dataset

plt.style.use('ggplot')
matplotlib.rcParams.update(
    {
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': True
    }
)

td = 2
numSamples = 3000000
availableData = 10000
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

originalData = np.load(config.ROOT_DIR + "data/{}_noisy_circle_samples.npy".format(numSamples))[:availableData, :]
# Could the difference in performance between MLP and TSM have to do with WHEN the larger score errors occur?
mlp_model = pickle.load(open(
    config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_TembDim{}_EncShapes{}".format(
        td,
        N, Tdiff, trainEps, temb_dim, enc_shapes), 'rb'))
tsm_model = pickle.load(open(
    config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(
        td,
        N, Tdiff, trainEps, temb_dim, residual_layers, residual_channels, diff_hidden_size), 'rb'))

# Run backward diffusion and record MSE error at each diffusion index AND plot samples for visualisation purposes
x_mlp = torch.randn((availableData, td)).to(torch.float32)
x_tsm = torch.randn((availableData, td)).to(torch.float32)
reverse_times = torch.linspace(start=Tdiff, end=sampleEps, steps=N)
dt = 1. / N
MLP_MSEs = np.empty(shape=(N,))
TSM_MSEs = np.empty(shape=(N,))
mlp_model.eval()
tsm_model.eval()
with torch.no_grad():
    for i in tqdm(range(0, N)):
        ts = reverse_times[i] * torch.ones((availableData, 1))

        z_mlp = torch.randn_like(x_mlp)
        mlp_score = mlp_model.model.forward(x_mlp, ts.squeeze(-1)).squeeze(1)
        mlp_true_score = -(x_mlp - torch.exp(-0.5 * ts) * originalData) / (1. - torch.exp(-ts))

        # Record error
        MLP_MSEs[i] = (torch.mean(torch.pow(mlp_score - mlp_true_score, 2.)).detach().numpy())
        # EM sampling
        x_mlp = x_mlp + (0.5 * x_mlp + mlp_score) * dt + np.sqrt(dt) * z_mlp

        z_tsm = torch.randn_like(x_tsm)

        tsm_score = tsm_model.model.forward(x_tsm, ts.squeeze(-1)).squeeze(1)
        tsm_true_score = -(x_tsm - torch.exp(-0.5 * ts) * originalData) / (1. - torch.exp(-ts))
        TSM_MSEs[i] = (torch.mean(torch.pow(tsm_score - tsm_true_score, 2.)).detach().numpy())

        x_tsm = x_tsm + (0.5 * x_tsm + tsm_score) * dt + np.sqrt(dt) * z_tsm

# Plot MSE errors
fig, ax = plt.subplots()
ax.plot(reverse_times, np.cumsum(np.sqrt(MLP_MSEs)) - np.cumsum(np.sqrt(TSM_MSEs)),
        label="Difference in Cumulative RMSEs")
ax.set_title("Comparison between MLP and TSM Cumulative RMSEs")
ax.set_xlabel("$\\textbf{Diffusion time}$")
ax.legend()
plt.show()

plot_dataset(originalData, x_mlp.detach().numpy())
plot_dataset(originalData, x_tsm.detach().numpy())
"""
# Plot current time samples every 50 indices
    if (i+1) % 1000 == 0 or i == 0:
        fig, ax = plt.subplots(1,2)
        ax[0].scatter(originalData[:,0], originalData[:, 1], alpha=0.6,label="Original Data")
        ax[0].scatter(x_mlp.detach().numpy()[:,0], x_mlp.detach().numpy()[:,1], alpha=0.3, label="MLP at Diff Index {}".format(N-1-i))
        ax[0].set_title("MLP")
        ax[0].set_aspect("equal", adjustable="box")
        ax[0].legend()
        ax[1].scatter(originalData[:, 0], originalData[:,1], alpha=0.6, label="Original Data")
        ax[1].scatter(x_tsm.detach().numpy()[:,0], x_mlp.detach().numpy()[:,1], alpha=0.3, label="TSM at Diff Index {}".format(N-1-i))
        ax[1].set_title("TSM")
        ax[1].set_aspect("equal", adjustable="box")
        ax[1].legend()
        plt.show()
        plt.close()
    """
