import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.math_functions import generate_fBm, chiSquared_test, fBm_to_fBn
from utils.plotting_functions import plot_diffusion_marginals

if __name__ == "__main__":
    h, td = 0.7, 2
    N = 100
    numSamples = 1000000
    rng = np.random.default_rng()

    data = generate_fBm(H=h, T=td, S=3000, rng=rng)
    trial_data = torch.from_numpy(data)

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

    reversed_timesteps = np.linspace(start=1., stop=1e-2, num=N)
    for i in tqdm((range(0, N))):
        z = torch.randn_like(x)
        dt = 1. / N
        beta_t = betas[(N - 1) - i]
        t = reversed_timesteps[i]

        # The noise is a function of the current sample!!
        noise = -(x - np.exp(-0.5 * beta_t * t) * trial_data) / (1. - np.exp(-beta_t * t))  # Score == Noise/STD!
        x = x + (0.5 * beta_t * x + beta_t * noise) * dt + np.sqrt(dt * beta_t) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]
    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(np.cov(true_samples.T)[0, 0],
                                                                     np.cov(true_samples.T)[0, 1]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(np.cov(generated_samples.T)[0, 0],
                                                                      np.cov(generated_samples.T)[0, 1]))
    print("Expected :: Var {} :: Covar {}".format(1., 0.5 * 2 ** 1.4 - 1.))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=fBm_to_fBn(generated_samples))
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

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
