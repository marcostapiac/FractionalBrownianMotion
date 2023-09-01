import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.data_processing import evaluate_SDE_performance
from utils.math_functions import generate_CEV
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    H, td = .7, 2
    muU = 1.
    muX = 2.
    alpha = 1.
    sigmaX = 0.5
    X0 = 1.
    U0 = 0.
    numSamples = 100000
    numDiffSteps = 1000
    rng = np.random.default_rng()

    trial_data = generate_CEV(H=H, T=td, S=numSamples, alpha=alpha, sigmaX=sigmaX, muU=muU, muX=muX, X0=X0, U0=U0,
                              rng=rng)
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

    evaluate_SDE_performance(true_samples, generated_samples,td=td)
