import numpy as np
import torch
from tqdm import tqdm

from utils.math_functions import generate_fBm
from utils.plotting_functions import compare_against_isotropic_Gaussian

""" This experiment aims to investigate how the convergence to a Gaussian is affected by the number of dimensions in the data """
if __name__ == "__main__":
    h, td = 0.7, 32
    N = 10000
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_fBm(H=h, T=td, S=numSamples, rng=rng)
    trial_data = torch.from_numpy(data).to(torch.float32)
    print(np.cov(trial_data, rowvar=False))

    # DDPM VP-SDE Parameters
    beta_min = 0.0001
    beta_max = 10.
    assert (beta_max < N)
    betas = torch.linspace(start=beta_min, end=beta_max, steps=N)/N # Discretised noise schedule
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
        if i>N/1000 and (i+1)%1000 == 0:
            print(i, i%1000, t)
            compare_against_isotropic_Gaussian(forward_samples=xts, td=td, rng=rng, diffTime=i+1)
            print(np.cov(xts, rowvar=False))
            for i in range(10000000): j = 0
        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())
