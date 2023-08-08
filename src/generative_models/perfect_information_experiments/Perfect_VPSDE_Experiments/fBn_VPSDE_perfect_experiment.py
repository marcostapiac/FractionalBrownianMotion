import numpy as np
import torch
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import generate_fBn, chiSquared_test, compute_fBn_cov
from utils.plotting_functions import plot_diffusion_marginals, plot_dataset

if __name__ == "__main__":
    h, td = 0.7, 2
    N = 100
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
    trial_data = torch.from_numpy(data).to(torch.float32)

    # DDPM VP-SDE Parameters
    beta_min = 0.0001
    beta_max = 10.
    assert (beta_max < N)
    betas = torch.linspace(start=beta_min, end=beta_max, steps=N) / N  # Discretised noise schedule
    alphas = torch.cumprod(1. - betas, dim=0)
    timesteps = torch.linspace(start=1e-5, end=1., steps=N)  # Discretised time axis

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
    sigNoiseRatio = 0.01
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

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(*np.cov(true_samples, rowvar=False)[0, :]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(*np.cov(generated_samples, rowvar=False)[0, :]))
    print("Expected :: Var {} :: Covar {}".format(*compute_fBn_cov(FractionalBrownianNoise(H=h, rng=rng), td)[0, :]))

    assert (np.cov(generated_samples.T)[0, 1] == np.cov(generated_samples.T)[1, 0])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=generated_samples)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    plot_dataset(true_samples, generated_samples)
