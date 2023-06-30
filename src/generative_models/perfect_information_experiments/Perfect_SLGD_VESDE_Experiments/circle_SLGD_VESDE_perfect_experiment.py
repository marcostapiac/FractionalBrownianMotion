import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
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
    numDiffSteps = 100
    numSamples = 50000
    rng = np.random.default_rng()

    data = generate_circles(T=td, S=numSamples)
    trial_data = torch.from_numpy(data)

    # Forward process for clarity only
    var_max = torch.Tensor([1. ** 2]).to(torch.float32)
    var_min = torch.Tensor([0.01 ** 2]).to(torch.float32)
    timesteps = torch.linspace(start=1e-5, end=1., steps=numDiffSteps).to(torch.float32)
    vars = var_min * torch.pow((var_max / var_min),
                               timesteps)  # SLGD defines noise sequence from big to small for REVERSE process, but this sequence corresponds to the FORWARD noise schedule

    sampless = []
    sampless.append(trial_data.numpy())
    for i in range(0, numDiffSteps):
        t = timesteps[i]
        x0s = trial_data.to(torch.float32)
        z = torch.randn_like(x0s)
        x = x0s + torch.sqrt(vars[i]) * z
        sampless.append(x.numpy())

    reversed_sampless = []
    x = torch.sqrt(var_max) * torch.randn_like(trial_data)  # Note the prior is NOT standard isotropic noise
    reversed_sampless.append(x.numpy())

    reversed_timesteps = torch.linspace(start=1., end=1e-3, steps=numDiffSteps).to(torch.float32)
    for i in tqdm(iterable=range(0, numDiffSteps), dynamic_ncols=False,
                  desc="Sampling :: ", position=0):
        z = torch.randn_like(x)
        dt = 1. / numDiffSteps
        t = reversed_timesteps[i]
        diffCoeffSqrd = var_min * torch.pow((var_max / var_min), t) * 2. * torch.log(var_max / var_min)
        score = -(x - trial_data) / (vars[(numDiffSteps - 1 - i)])
        x = x + (diffCoeffSqrd * score) * dt + torch.sqrt(dt * diffCoeffSqrd) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

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
