import pickle

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import datasets
from tqdm import tqdm

from utils import config
from utils.plotting_functions import plot_final_diffusion_marginals, plot_dataset, qqplot

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
fig, ax = plt.subplots()

td = 2
Tdiff = 1.
N = 1000
trainEps = 1e-3
data = np.load(config.ROOT_DIR + "data/a_million_noisy_circle_samples_T{}.npy".format(td))
diffusion = pickle.load(open(
    config.ROOT_DIR + "src/generative_models/models/trained_noisy_circle_OU_model_T{}_Ndiff{}_Tdiff{}_trainEps{:.0e}".format(
        td,
        N, int(Tdiff), trainEps),
    'rb'))

algData = data[:1000]
fwd_samples = [algData]
fwd_times = np.linspace(trainEps, Tdiff, N)
for _ in range(N):
    t = fwd_times[_]
    sample_at_t = np.exp(-0.5 * t) * algData + np.sqrt(1 - np.exp(-t)) * np.random.normal(size=algData.shape)
    fwd_samples.append(sample_at_t)

x = torch.randn(size=algData.shape).to(torch.float32)
#x = torch.exp(-torch.Tensor([0.5])) * torch.from_numpy(algData).to(torch.float32) + torch.sqrt(1. - torch.exp(torch.Tensor([-1]))) * x
bkwd_samples = [x.numpy()] * (N + 1)
dt = 1. / N
reverseTimes = np.linspace(start=Tdiff, stop=trainEps, num=N)
with torch.no_grad():
    for i in tqdm(iterable=(range(0, N)), dynamic_ncols=False,
                  desc="Sampling :: ", position=0):
        ts = reverseTimes[i] * np.ones((algData.shape[0], 1))  # time-index for each data-sample
        predicted_score = diffusion.model.forward(x, torch.from_numpy(ts).to(torch.float32).squeeze(-1)).squeeze(
            1)  # Score == Noise/STD!
        z = torch.randn_like(x)
        x = x + (0.5 * x + predicted_score) * dt + np.sqrt(dt) * z
        bkwd_samples[N - 1 - i] = x.numpy()

# Now plot for each time, the corresponding fwd and backwd samples, in reverse-time order:
for i in range(N + 1):
    if i % 100 == 0 and i > 800:
        fwd_sample = fwd_samples[N - i]
        bkwd_sample = bkwd_samples[N - i]
        fig, ax = plt.subplots(1, 2)
        ax[0].scatter(fwd_sample[:, 0], fwd_sample[:, 1])
        ax[0].scatter(algData[:, 0], algData[:, 1])
        ax[0].grid(False)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_title("Forward Sample at index {}".format(N - i))
        ax[1].scatter(bkwd_sample[:, 0], bkwd_sample[:, 1])
        ax[1].scatter(algData[:, 0], algData[:, 1])
        ax[1].grid(False)
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_title("Backward Sample at index {}".format(N - i))
        plt.show()

plot_dataset(algData, bkwd_samples[0])
plot_final_diffusion_marginals(algData, bkwd_samples[0], timeDim=td)

# Check if issue is still present if we compare samples from same datasets
qqplot(x=algData[:, 0],
       y=data[30000:40000, 0], xlabel="First {} Data Samples".format(10000),
       ylabel="Another {} Data Samples".format(10000),
       plottitle="Marginal Q-Q Plot", log=False)
plt.show()
plt.close()

## Check if issue is reduced if dataset is drawn from a probabilistic distribution
X, y = datasets.make_circles(
    n_samples=100000, noise=0.03, random_state=None, factor=.5)
sample = X * 4
# Compare deterministic dataset with noisy one
plot_dataset(algData, sample)
qqplot(x=sample[:10000, 0],
       y=sample[30000:40000, 0], xlabel="First {} Data Samples".format(10000),
       ylabel="Another {} Data Samples".format(10000),
       plottitle="Marginal Q-Q Plot", log=False)
plt.show()
plt.close()

# Compare noisy distribution with backward samples
plot_dataset(sample, bkwd_samples[0])
qqplot(x=sample,
       y=bkwd_samples[0], xlabel="Noisy Dataset Samples",
       ylabel="Reverse Samples at Diff Time {}".format(0),
       plottitle="Marginal Q-Q Plot at Time Dim {}".format(0), log=False)
plt.show()
plt.close()

# Is the problem the QQ plot is very sensitive to outliers? What if we remove them?
prunedData = []
for datapoint in bkwd_samples[0]:
    assert (datapoint.shape[0] == 2)
    r = np.sqrt(datapoint[0] ** 2 + datapoint[1] ** 2)
    if 1.95 <= r <= 2.05 or 3.95 <= r <= 4.05:
        prunedData.append(datapoint)
prunedData = np.array(prunedData)
idxs = np.random.randint(0, algData.shape[0], size=algData.shape[0] - prunedData.shape[0])
prunedAlgData = np.delete(algData, idxs, axis=0)
plot_dataset(prunedAlgData, prunedData)
plot_final_diffusion_marginals(prunedAlgData, prunedData, timeDim=td)

# Is the problem the distribution of points in each of the concetric circles?
innerb = 0
outerb = 0
innerf = 0
outerf = 0
S = algData.shape[0]
for i in range(S):
    bkwd = bkwd_samples[0][i]
    fwd = algData[i]
    rb = np.sqrt(bkwd[0] ** 2 + bkwd[1] ** 2)
    rf = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
    if rb <= 2.1:
        innerb += 1
    elif 3.9 <= rb:
        outerb += 1
    if rf <= 2.1:
        innerf += 1
    elif 3.9 <= rf:
        outerf += 1

print("Generated: Inner {} vs Outer {}".format(innerb / S, outerb / S))
print("True: Inner {} vs Outer {}".format(innerf / S, outerf / S))

## It seems like generated samples have a larger concentration of points in inner circle, compared to outer one...