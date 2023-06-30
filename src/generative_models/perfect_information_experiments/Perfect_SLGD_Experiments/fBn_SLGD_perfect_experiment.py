import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.math_functions import chiSquared_test, generate_fBn
from utils.plotting_functions import plot_diffusion_marginals

if __name__ == "__main__":
    h, td = 0.7, 2
    numLangSteps = 1000
    numDiffSteps = 10
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
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

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(np.cov(true_samples.T)[0, 0],
                                                                     np.cov(true_samples.T)[0, 1]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(np.cov(generated_samples.T)[0, 0],
                                                                      np.cov(generated_samples.T)[0, 1]))
    print("Expected :: Var {} :: Covar {}".format(1., 0.5 * 2 ** 1.4 - 1.))
    assert (np.cov(generated_samples.T)[0, 1] == np.cov(generated_samples.T)[1, 0])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=generated_samples)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

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
