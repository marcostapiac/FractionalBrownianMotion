import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.math_functions import generate_CEV

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    H, td = .7, 256
    muU = 1.
    muX = 2.
    alpha = 1.
    sigmaX = 0.5
    X0 = 1.
    U0 = 0.
    numSamples = 10000
    numDiffSteps = 100
    rng = np.random.default_rng()

    trial_data = generate_CEV(H=H, T=td, S=numSamples, alpha=alpha, sigmaX=sigmaX, muU=muU, muX=muX, X0=X0, U0=U0,
                              rng=rng)

    trial_data = torch.from_numpy(trial_data)

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
        t = reversed_timesteps[i]
        dt = 1. / numDiffSteps
        diffCoeffSqrd = var_min * torch.pow((var_max / var_min), t) * 2. * torch.log(var_max / var_min) * dt
        score = -(x - trial_data) / (var_min * torch.pow((var_max / var_min), t))
        x = x + (diffCoeffSqrd * score) + torch.sqrt(diffCoeffSqrd) * z
        for j in range(10):
            z = torch.randn_like(x)
            e = 2 * (0.1 * np.linalg.norm(z) / np.linalg.norm(score)) ** 2
            x = x + e * score + np.sqrt(2. * e) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    evaluate_SDE_HigherDim_performance(trial_data.numpy(), reversed_sampless[-1], td=td)
