import itertools

import numpy as np
import torch

from utils import config
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
fig, ax = plt.subplots()

td = 2
data = np.load(config.ROOT_DIR + "data/{}_noisy_circle_samples.npy".format(102890))[0, :].reshape((td,1))
eps = 1e-3
Tdiff = [1.,]
N = [1000, 10000]
trialPairs = [*itertools.product(Tdiff, N)]
trialPairs = [(1., 500), (1., 1000), (1.,2000)]


for Tdiff, Ndiff in trialPairs:
#Tdiff, Ndiff = 10., 1000
#beta_min = 0.1
#beta_max = 20.
#sigma_min = 0.1
#sigma_max = 9.
#for i in range(3):
# VPtimes = 0.5*OUtimes**2*(beta_max-beta_min)+OUtimes*beta_min
#VEtimes = sigma_min**2*(sigma_max/sigma_min)**(2.*OUtimes)
    effTimes = np.linspace(start=eps, stop=Tdiff, num=Ndiff)
    # Compute KL divergence
    vars0 = (1. - np.exp(-effTimes[1:]))
    vars1 = (1. - np.exp(-effTimes[:Ndiff - 1]))
    means0 = np.exp(-0.5 * effTimes[1:])
    means1 = np.exp(-0.5 * effTimes[:Ndiff - 1])
    # Compute terms independently
    traceTerms = td * vars0/vars1
    logTerm = td*np.log(vars1/vars0)
    distanceTerm = ((means1-means0)*(means1-means0)/vars1)*np.squeeze(data.T@data)
    KLs = 0.5*(traceTerms+logTerm+distanceTerm-td)
    # Indices of time (starting with ti)
    indices = np.linspace(1, Ndiff-1, Ndiff-1)
    ax.plot(indices[:10], KLs[:10], label="T, N : {}, {}".format(Tdiff, Ndiff))

ax.set_title("KL Divergence Between")
ax.set_xlabel("Diffusion Index")
ax.set_yscale("log")
ax.legend()
plt.show()

