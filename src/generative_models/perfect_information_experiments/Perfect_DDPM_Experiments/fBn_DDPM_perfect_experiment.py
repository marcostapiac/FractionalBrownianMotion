import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.classes.ClassDenoisingDiffusion import DenoisingDiffusion
from src.classes.ClassTimeSeriesNoiseMatching import TimeSeriesNoiseMatching
from utils.math_functions import generate_fBn, chiSquared_test
from utils.plotting_functions import plot_diffusion_marginals

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
if __name__ == "__main__":
    h, td = 0.7, 2
    N = 10
    numSamples = 1000000
    rng = np.random.default_rng()

    scoreModel = TimeSeriesNoiseMatching()
    model = DenoisingDiffusion(device="cpu", model=scoreModel, N=N, rng=rng)

    data = generate_fBn(H=h, T=td, S=3000, rng=rng)
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
        noise = ((x - np.sqrt(model.alphaBars[t]) * trial_data) / np.sqrt(1. - model.alphaBars[t]))
        x = model.postCoeff1[t] * x - model.postCoeff2[t] * noise + torch.sqrt(model.reverseVars[t]) * z
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

    plot_diffusion_marginals(true_samples, generated_samples, timeDim=td, diffTime=0)
