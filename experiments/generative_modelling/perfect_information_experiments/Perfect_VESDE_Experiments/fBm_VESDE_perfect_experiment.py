import numpy as np
import torch
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import chiSquared_test, reduce_to_fBn, generate_fBm, compute_fBm_cov
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset

if __name__ == "__main__":
    h, td = 0.7, 2
    numDiffSteps = 100
    numSamples = 10000
    rng = np.random.default_rng()

    data = generate_fBm(H=h, T=td, S=numSamples, rng=rng)
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
        t = reversed_timesteps[i]
        dt = 1. / numDiffSteps
        diffCoeffSqrd = var_min * torch.pow((var_max / var_min), t) * 2. * torch.log(var_max / var_min) * dt
        score = -(x - trial_data) / (var_min * torch.pow((var_max / var_min), t))
        x = x + (diffCoeffSqrd * score) + torch.sqrt(diffCoeffSqrd) * z
        for j in range(10):
            z = torch.randn_like(x)
            e = 2 * (0.1 * np.linalg.norm(z) / np.linalg.norm(g)) ** 2
            x = x + e * score + np.sqrt(2. * e) * z
        reversed_sampless.append(x.numpy())

    true_samples = sampless[0]
    generated_samples = reversed_sampless[-1]

    print("Original Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(true_samples, axis=0)))
    print("Generated Data Sample Mean :: Dim 1 {} :: Dim 2 {}".format(*np.mean(generated_samples, axis=0)))
    print("Expected Mean :: Dim 1 {} :: Dim 2 {}".format(0., 0.))
    print("Original Data :: \n [[{}, {}]\n[{},{}]]".format(*np.cov(true_samples, rowvar=False).flatten()))
    print("Generated Data :: \n [[{}, {}]\n[{},{}]]".format(*np.cov(generated_samples, rowvar=False).flatten()))
    print("Expected :: \n [[{}, {}]\n[{},{}]]".format(
        *compute_fBm_cov(FractionalBrownianNoise(H=h, rng=rng), td=td).flatten()))

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=td, H=h, samples=reduce_to_fBn(generated_samples))
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_dataset(true_samples, generated_samples)

    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)
