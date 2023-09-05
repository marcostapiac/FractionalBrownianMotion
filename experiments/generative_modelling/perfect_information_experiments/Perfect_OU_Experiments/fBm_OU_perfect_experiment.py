import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import generate_fBm, chiSquared_test, reduce_to_fBn, compute_fBm_cov
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    h = 0.7
    td = 2
    N = 1000  # In OU, constant noise schedule implies need for longer diffusion chain (try with 100)
    numSamples = 10000
    rng = np.random.default_rng()
    data = generate_fBm(H=h, T=td, S=numSamples, rng=rng)

    trial_data = data

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(trial_data[:, 0], trial_data[:, 1], alpha=0.6)
    ax.grid(False)
    ax.set_aspect("equal")
    strtitle = "Scatter Plot of 2-dimensional fBm"
    ax.set_title(strtitle)
    ax.set_xlabel("$\\textbf{Time Dim 1}$")
    ax.set_ylabel("$\\textbf{Time Dim 2}$")
    plt.legend()
    plt.show()

    # Forward sampling only shown for clarity, not needed
    ts = np.linspace(start=1e-3, stop=.1, num=N)  # Empirically better samples with 1e-3
    noises = []
    sampless = []
    sampless.append(trial_data)

    for t in ts:
        x0s = trial_data
        epsts = rng.normal(size=x0s.shape)

        xts = np.exp(-0.5 * t) * x0s + np.sqrt((1. - np.exp(-t))) * epsts

        noises.append(epsts)
        sampless.append(xts)

    reversed_sampless = []
    x = rng.normal(size=trial_data.shape)
    reversed_sampless.append(x)

    timesteps = np.linspace(.1, stop=1e-3, num=N)  # Empirically better samples with 1e-3

    for i in tqdm((range(N))):
        dt = 1. / N
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
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: \n [[{}, {}]\n[{},{}]]".format(*np.cov(true_samples, rowvar=False).flatten()))
    print("Generated Data :: \n [[{}, {}]\n[{},{}]]".format(*np.cov(generated_samples, rowvar=False).flatten()))
    print("Expected :: \n [[{}, {}]\n[{},{}]]".format(
        *compute_fBm_cov(FractionalBrownianNoise(H=h, rng=rng), td=td, isUnitInterval=True).flatten()))

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=reduce_to_fBn(generated_samples), isUnitInterval=True)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_dataset(true_samples, generated_samples)

    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)
