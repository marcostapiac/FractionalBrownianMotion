import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.math_functions import generate_circles

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 10
    numSamples = 10000
    rng = np.random.default_rng()
    data = generate_circles(T=td, S=numSamples)

    trial_data = torch.from_numpy(data).to(torch.float32)

    # DDPM VP-SDE Parameters
    beta_min = 0.1
    beta_max = 20.
    betas = torch.linspace(start=beta_min, end=beta_max, steps=N)  # Discretised noise schedule
    timesteps = torch.linspace(start=1e-3, end=1., steps=N)  # Discretised time axis

    sampless = []
    sampless.append(trial_data.numpy())
    for i in tqdm(range(N)):
        t = timesteps[i]

        x0s = trial_data.to(torch.float32)
        epsts = torch.randn_like(x0s)
        beta_t = betas[i]  # Noise at time ts[i]
        xts = torch.exp(-0.5 * beta_t * t) * x0s + torch.sqrt(1. - np.exp(-beta_t * t)) * epsts
        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())

    reversed_timesteps = torch.linspace(start=1., end=1e-1, steps=N)
    for i in tqdm((range(0, N))):
        z = torch.randn_like(x)
        dt = 1. / N
        beta_t = betas[(N - 1) - i]
        t = reversed_timesteps[i]

        # The noise is a function of the current sample!!
        noise = -(x - np.exp(-0.5 * beta_t * t) * trial_data) / (1. - np.exp(-beta_t * t))  # Score == Noise/STD!
        x = x + (0.5 * beta_t * x + beta_t * noise) * dt + np.sqrt(dt * beta_t) * z
        reversed_sampless.append(x.numpy())

    true_samples = trial_data
    generated_samples = reversed_sampless[-1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Time Dim 1")
    ax.set_ylabel("Time Dim 2")
    plt.legend()
    plt.show()
