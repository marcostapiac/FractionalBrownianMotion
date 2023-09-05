import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import generate_fBn, chiSquared_test, compute_fBn_cov
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
    N = 1000  # In OU, constant noise schedule implies need for longer diffusion chain (try with 10) (TODO WHY? -> it does not have to do with convergence to prior)
    numSamples = 100000
    rng = np.random.default_rng()
    data = generate_fBn(H=h, T=td, S=numSamples, rng=rng)

    trial_data = data
    ts = np.linspace(start=1e-3, stop=1., num=N)  # Empirically better samples with 1e-3
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

    timesteps = np.linspace(1., stop=1e-3,
                            num=N)  # Best epsilon (stop) depends on the number of diffusion steps (1.2e-4 for 100, 1e-5 for 1000)! (TODO WHY?)

    for i in tqdm((range(N))):
        dt = 1. / N
        t = timesteps[i]
        z = rng.normal(size=x.shape)
        g = - (x - np.exp(-0.5 * t) * trial_data) / (1. - np.exp(-t))
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z
        # TODO: Corrector step for improved sample quality
        reversed_sampless.append(x)

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

    plot_dataset(true_samples, generated_samples)

    plot_final_diffusion_marginals(true_samples, generated_samples, timeDim=td)
