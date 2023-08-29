import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.math_functions import generate_circles
from utils.plotting_functions import qqplot

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 1000  # In OU, constant noise schedule implies need for longer diffusion chain (try with 100)
    numSamples = 100000
    rng = np.random.default_rng()
    data = generate_circles(T=td, S=numSamples)

    trial_data = data

    # Visualise data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(trial_data[:, 0], trial_data[:, 1], alpha=0.6, label=None)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Circle Dataset"
    ax.set_title(strtitle)
    ax.set_xlabel("Time Dim 1")
    ax.set_ylabel("Time Dim 2")
    plt.legend()
    plt.show()

    ts = np.linspace(start=1e-6, stop=N,
                     num=N)  # Instability as epsilon gets small (best epsilon varies is data-driven)
    noises = []
    sampless = []
    sampless.append(trial_data)

    x0s = trial_data
    for i in tqdm(range(N)):
        t = ts[i]
        epsts = np.random.randn(*x0s.shape)

        xts = np.exp(-0.5 * t) * x0s + np.sqrt((1. - np.exp(-t))) * epsts

        noises.append(epsts)
        sampless.append(xts)

    print(np.mean(sampless[-1], axis=0),np.cov(sampless[-1], rowvar=False))
    normal = rng.multivariate_normal([0.,0.], np.eye(2) ,sampless[-1].shape[0])
    gensamples = sampless[-1]
    qqplot(sampless[-1], normal)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(normal[:, 0], normal[:, 1], alpha=0.6, label="Original Data")
    ax.scatter(gensamples[:, 0], gensamples[:, 1], alpha=0.2, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

    reversed_sampless = []
    x = np.random.randn(*trial_data.shape)
    reversed_sampless.append(x)

    timesteps = np.linspace(1, stop=1e-3, num=N)

    for i in tqdm((range(N))):
        dt = 1. / N
        t = timesteps[i]
        z = np.random.randn(*x.shape)
        g = - (x - np.exp(-0.5 * t) * trial_data) / (1. - np.exp(-t))
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z  # Predictor
        reversed_sampless.append(x)

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]
    print(np.mean(true_samples, axis=0), np.cov(true_samples, rowvar=False))
    print(np.mean(generated_samples, axis=0), np.cov(generated_samples, rowvar=False))
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
