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
    N = 1000  # In OU, constant noise schedule implies need for longer diffusion chain (does it?)
    numSamples = 10000
    rng = np.random.default_rng()
    latent = torch.from_numpy(generate_circles(T=td, S=numSamples))
    z0 = torch.randn_like(latent)
    observed = latent + 6. * torch.ones_like(latent) + np.sqrt(0.001) * z0

    trial_data = latent

    ts = torch.linspace(start=1e-3, end=1.,
                        steps=N)  # Instability as epsilon gets small (best epsilon varies is data-driven)

    sampless = []
    sampless.append(latent.numpy())

    for t in ts:
        epsts = torch.randn_like(trial_data)
        xts = torch.exp(-0.5 * t) * trial_data + torch.sqrt((1. - torch.exp(-t))) * epsts
        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())

    timesteps = torch.linspace(1., end=1e-3, steps=N)

    for i in tqdm((range(N))):
        dt = 1. / N
        t = timesteps[i]
        z = torch.randn_like(trial_data)
        latent_score = - (x - torch.exp(-0.5 * t) * trial_data) / (1. - torch.exp(-t))
        x0_hat = np.exp(0.5 * t) * x - np.sqrt(np.exp(t) - 1.) * torch.randn_like(x)
        observed_score = np.exp(0.5 * t) * (observed - (x0_hat + 6. * torch.ones_like(latent))) / (
            0.01)  # grad log p(y0|xt)
        g = latent_score + observed_score
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z  # Predictor
        # TODO: Corrector!
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = (reversed_sampless[-1])

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
