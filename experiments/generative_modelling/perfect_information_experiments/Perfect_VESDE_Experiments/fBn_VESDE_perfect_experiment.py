import numpy as np
import torch
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import chiSquared_test, generate_fBn, compute_fBn_cov
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset

if __name__ == "__main__":
    h, td = 0.7, 2
    numDiffSteps = 100
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)
    trial_data = torch.from_numpy(data).to(torch.float32)

    # Forward process for clarity only
    var_max = torch.Tensor([1. ** 2]).to(torch.float32)
    var_min = torch.Tensor([0.01 ** 2]).to(torch.float32)
    indices = torch.linspace(start=0, end=numDiffSteps, steps=numDiffSteps)
    vars = var_min * torch.pow((var_max / var_min),
                               indices / (
                                       numDiffSteps - 1))  # SLGD defines noise sequence from big to small for REVERSE process, but this sequence corresponds to the FORWARD noise schedule
    sampless = []
    sampless.append(trial_data.numpy())
    timesteps = torch.linspace(start=1e-3, end=1., steps=numDiffSteps).to(torch.float32)
    for i in range(0, numDiffSteps):
        t = timesteps[i]
        z = torch.randn_like(trial_data)
        var_t = var_min * (var_max / var_min) ** t
        x = trial_data + torch.sqrt(var_t) * z
        sampless.append(x.numpy())

    reversed_sampless = []
    x = torch.sqrt(var_max) * torch.randn_like(trial_data)  # Note the prior is NOT standard isotropic noise
    reversed_sampless.append(x.numpy())

    reversed_timesteps = torch.linspace(start=1., end=1e-3, steps=numDiffSteps).to(torch.float32)
    for i in tqdm(iterable=range(0, numDiffSteps), dynamic_ncols=False,
                  desc="Sampling :: ", position=0):
        z = torch.randn_like(x)
        t = reversed_timesteps[i]
        # dt = 1. / numDiffSteps
        # diffCoeffSqrd = var_min * torch.pow((var_max / var_min), t) * 2. * torch.log(var_max / var_min)
        # x = x + (diffCoeffSqrd * score) * dt + torch.sqrt(dt * diffCoeffSqrd) * z
        for j in range(10):
            z = torch.randn_like(x)
            g = -(x - trial_data) / (var_min * torch.pow((var_max / var_min), t))
            e = 2 * (0.1 * np.linalg.norm(z) / np.linalg.norm(g)) ** 2
            x = x + e * g + np.sqrt(2. * e) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: Sample Var {} :: Sample Covar {}".format(*np.cov(true_samples, rowvar=False)[0, :]))
    print("Generated Data :: Sample Var {} :: Sample Covar {}".format(*np.cov(generated_samples, rowvar=False)[0, :]))
    print("Expected :: Var {} :: Covar {}".format(*compute_fBn_cov(FractionalBrownianNoise(H=h, rng=rng), td)[0, :]))

    assert (np.cov(generated_samples.T)[0, 1] == np.cov(generated_samples.T)[1, 0])

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=generated_samples, isUnitInterval=True)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)

    plot_dataset(true_samples, generated_samples)