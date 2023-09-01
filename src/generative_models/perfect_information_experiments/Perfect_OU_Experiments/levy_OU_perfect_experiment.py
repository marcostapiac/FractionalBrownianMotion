import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from levy_processes.mean_mixture_processes import NormalGammaProcess
from utils.plotting_functions import plot_final_diffusion_marginals

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def simulate_NG_samples(T: int, S: int, rng: np.random.Generator) -> np.array:
    mu_W = 1.0
    var_W = 2.0
    beta = .5
    C = 1.0
    process = NormalGammaProcess(beta=beta, C=C, mu=0., mu_W=mu_W, var_W=var_W, rng=rng)
    X = np.zeros(shape=(S, T))
    for i in tqdm(range(S)):
        X[i, :] = process.simulate_path(observation_times=np.arange(0., 1., step=1. / (T), dtype=float))
    return X


if __name__ == "__main__":
    H, td = .7, 2
    muU = 1.
    muX = 2.
    alpha = 1.
    sigmaX = 0.7
    X0 = 1.
    U0 = 0.
    numSamples = 10000
    numDiffSteps = 1000
    rng = np.random.default_rng()

    trial_data = simulate_NG_samples(T=td, S=numSamples, rng=rng)
    print(trial_data)
    # Forward sampling only shown for clarity, not needed
    ts = np.linspace(start=1e-3, stop=1., num=numDiffSteps)  # Empirically better samples with 1e-3

    sampless = []
    sampless.append(trial_data)

    for t in ts:
        x0s = trial_data
        epsts = rng.normal(size=x0s.shape)

        xts = np.exp(-0.5 * t) * x0s + np.sqrt((1. - np.exp(-t))) * epsts

        sampless.append(xts)

    reversed_sampless = []
    x = rng.normal(size=trial_data.shape)
    reversed_sampless.append(x)

    timesteps = np.linspace(1., stop=1e-3, num=numDiffSteps)  # Empirically better samples with 1e-3

    for i in tqdm((range(numDiffSteps))):
        dt = 1. / numDiffSteps
        t = timesteps[i]
        z = rng.normal(size=x.shape)
        g = - (x - np.exp(-0.5 * t) * trial_data) / (1. - np.exp(-t))
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z
        # TODO: Corrector step for improved sample quality
        reversed_sampless.append(x)

    true_samples = trial_data
    generated_samples = reversed_sampless[-1]

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(np.cov(true_samples.T)[0, 0],
                                                                     np.cov(true_samples.T)[0, 1]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(np.cov(generated_samples.T)[0, 0],
                                                                      np.cov(generated_samples.T)[0, 1]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    # ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    # ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Time Dim 1")
    ax.set_ylabel("Time Dim 2")
    plt.legend()
    plt.show()

    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=2)
