import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from configs import project_config

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})
fig, ax = plt.subplots()

td = 2
data = np.load(project_config.ROOT_DIR + "data/noisy_circle_samples.npy")[0, :].reshape((td, 1))
eps = 1e-4

Tmax = 1.2
S = int(Tmax / 1e-3)
times = np.linspace(eps, Tmax, num=S)
for i in range(3):
    if i == 0:
        # OU SDE
        vars0 = (1. - np.exp(-times))
        means0 = np.exp(-0.5 * times.reshape((S, 1))) @ data.T
        vars1 = 0. * vars0 + 1.
        label = "OU Diffusion Model"
    elif i == 1:
        # VP SDE
        beta_min = 0.1
        beta_max = 20.
        effTimes = (0.5 * times ** 2 * (beta_max - beta_min) + times * beta_min)
        vars0 = (1. - np.exp(-effTimes))
        means0 = np.exp(-0.5 * effTimes.reshape((S, 1))) @ data.T
        vars1 = 0. * vars0 + 1.
        label = "VP Diffusion Model"
    else:
        # VE SDE
        std_min = np.sqrt(0.1)
        std_max = np.sqrt(40.)
        vars0 = std_min ** 2 * (std_max / std_min) ** (2. * times)
        means0 = np.vstack([data] * S).reshape((S, td))
        vars1 = 0. * vars0 + np.power(std_max, 2.)
        label = "VE Diffusion Model"
    # Compute terms independently
    means1 = np.zeros(shape=(S, td))
    traceTerms = td * vars0 / vars1
    logTerm = td * np.log(vars1 / vars0)
    distanceTerm = (np.sum(np.power((means1 - means0) * (means1 - means0), 2), axis=1)) / vars1
    KLs = 0.5 * (traceTerms + logTerm + distanceTerm - td)
    ax.plot(times, KLs, label=label)

ax.set_xlabel("$\\textbf{Terminal Diffusion Time } \\boldsymbol{T_{d}}$")
ax.set_title("Terminal KL Divergence")
ax.set_yscale("log")
ax.legend()
plt.show()
