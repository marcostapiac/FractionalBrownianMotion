import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.data_processing import evaluate_circle_performance
from utils.math_functions import generate_circles

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 10000
    numSamples = 10000
    rng = np.random.default_rng()
    data = generate_circles(T=td, S=numSamples)

    trial_data = torch.from_numpy(data).to(torch.float32)

    # DDPM VP-SDE Parameters
    beta_min = 0.0001
    beta_max = 10.
    betas = torch.linspace(start=beta_min, end=beta_max, steps=N) / N  # Discretised noise schedule
    alphas = torch.cumprod(1. - betas, dim=0)
    timesteps = torch.linspace(start=1e-3, end=1., steps=N)  # Discretised time axis

    sampless = []
    sampless.append(trial_data.numpy())
    for i in tqdm(range(N)):
        t = timesteps[i]

        x0s = trial_data.to(torch.float32)
        epsts = torch.randn_like(x0s)
        beta_int = (0.5 * t ** 2 * (beta_max - beta_min) + t * beta_min)  # Noise integral at time ts[i]
        xts = torch.exp(-0.5 * beta_int) * x0s + torch.sqrt(1. - np.exp(-beta_int)) * epsts
        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())

    reversed_timesteps = torch.linspace(start=1., end=1e-3, steps=N)
    sigNoiseRatio = 0.1
    for i in tqdm((range(0, N))):
        t = reversed_timesteps[i]
        beta_int = (0.5 * t ** 2 * (
                beta_max - beta_min) + t * beta_min)
        beta_t = betas[N - 1 - i]  # dt absorbed already
        score = -(x - np.exp(-0.5 * beta_int) * trial_data) / (1. - np.exp(-beta_int))
        z = torch.randn_like(x)
        x = x * (1. + 0.5 * beta_t) + beta_t * score + np.sqrt(beta_t) * z
        for _ in range(10):
            z = torch.randn_like(x)
            e = 2. * alphas[N - 1 - i] * (
                    sigNoiseRatio * np.linalg.norm(z) / np.linalg.norm(score)) ** 2
            x = x + e * score + np.sqrt(2. * e) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    evaluate_circle_performance(true_samples, generated_samples, td=td)
