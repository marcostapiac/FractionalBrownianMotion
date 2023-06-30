import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from tueplots import bundles

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils.math_functions import generate_circles

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 1000
    numSamples = 10000
    rng = np.random.default_rng()
    data = generate_circles(T=td, S=numSamples)

    scoreModel = TimeSeriesNoiseMatching()
    model = DenoisingDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)

    trial_data = torch.from_numpy(data)

    ts = torch.arange(start=0, end=N, step=1)
    sampless = []
    sampless.append(trial_data.numpy())

    for i in tqdm(range(N)):
        t = ts[i]

        x0s = trial_data.to(torch.float64)
        epsts = torch.randn_like(x0s)

        xts = np.sqrt(model.alphaBars[t]) * x0s + np.sqrt(1. - model.alphaBars[t]) * epsts

        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())

    for t in tqdm(reversed(range(0, N))):
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        # Even if noise is learnt perfectly, there exists error accumulation due to x_t_back != x_t_fwd
        noise = ((x - np.sqrt(model.alphaBars[t]) * trial_data) / np.sqrt(
            1. - model.alphaBars[t]))  # Note score = noise/std of forward!
        x = model.postCoeff1[t] * x - model.postCoeff2[t] * noise + torch.sqrt(model.reverseVars[t]) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    plt.rcParams.update(bundles.neurips2022())
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(generated_samples[:1000, 0], generated_samples[:1000, 1], alpha=0.3, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend()
    plt.show()
