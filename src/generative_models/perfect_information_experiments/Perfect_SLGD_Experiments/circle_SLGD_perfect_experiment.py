import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.math_functions import generate_circles
from utils.plotting_functions import plot_diffusion_marginals

if __name__ == "__main__":
    td = 2
    numLangSteps = 100
    numDiffSteps = 10
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_circles(T=td, S=numSamples)
    trial_data = torch.from_numpy(data)

    # Forward process for clarity only
    sigma_start = 1.
    sigma_end = 0.01
    vars = torch.pow(torch.exp(
        torch.linspace(np.log(sigma_start), np.log(sigma_end), numDiffSteps, dtype=torch.float32, requires_grad=False)),
        2.)

    sampless = []
    sampless.append(trial_data.numpy())
    for i in range(0, numDiffSteps):
        x0s = trial_data.to(torch.float32)
        z = torch.randn_like(x0s)
        x = x0s + torch.sqrt(
            vars[(numDiffSteps - 1) - i]) * z  # Note vars are defined in terms of the reverse process for SLGD
        sampless.append(x.numpy())

    reversed_sampless = []
    x = torch.sqrt(vars[0]) * torch.randn_like(trial_data)  # Note the prior is not standard isotropic noise
    reversed_sampless.append(x.numpy())
    alphas = 2e-5 * vars / vars[-1]

    for i in tqdm(iterable=range(0, numDiffSteps), dynamic_ncols=False,
                  desc="Sampling :: ", position=0):
        alpha = alphas[i]
        for _ in (range(0, numLangSteps)):
            z = torch.randn_like(x)
            predicted_score = - (x - trial_data) / vars[i]
            x = x + 0.5 * alpha * predicted_score + torch.sqrt(alpha) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, label="Original Data")
    ax.scatter(generated_samples[:1000, 0], generated_samples[:1000, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Time Dim 1")
    ax.set_ylabel("Time Dim 2")
    plt.legend()
    plt.show()
