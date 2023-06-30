import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.math_functions import generate_circles

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 1000  # In OU, constant noise schedule implies need for longer diffusion chain (try with 100)
    numSamples = 10000
    rng = np.random.default_rng()
    data = generate_circles(T=td, S=numSamples)

    trial_data = data

    ts = np.linspace(start=1e-6, stop=1.,
                     num=N)  # Instability as epsilon gets small (best epsilon varies is data-driven)
    noises = []
    sampless = []
    sampless.append(trial_data)

    for t in ts:
        x0s = trial_data
        epsts = np.random.randn(*x0s.shape)

        xts = np.exp(-0.5 * t) * x0s + np.sqrt((1. - np.exp(-t))) * epsts

        noises.append(epsts)
        sampless.append(xts)

    reversed_sampless = []
    x = np.random.randn(*trial_data.shape)
    reversed_sampless.append(x)

    timesteps = np.linspace(1., stop=1e-3, num=N)

    for i in tqdm((range(N))):
        dt = 1. / N
        t = timesteps[i]
        z = np.random.randn(*x.shape)
        g = - (x - np.exp(-0.5 * t) * trial_data) / (1. - np.exp(-t))
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z
        reversed_sampless.append(x)

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(generated_samples[:1000, 0], generated_samples[:1000, 1], alpha=0.3, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend()
    plt.show()
