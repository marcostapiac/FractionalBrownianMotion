import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils.plotting_functions import plot_final_diffusion_marginals

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

if __name__ == "__main__":
    td = 2
    N = 1000  # In OUSDE, constant noise schedule implies need for longer diffusion chain (does it?)
    numSamples = 2000
    rng = np.random.default_rng()
    muX = [10., 20.]
    covX = np.linalg.inv(np.array([[.5, .2], [.2, .5]]))
    covY = np.linalg.inv(np.array([[.5, -0.2], [-0.2, .5]]))
    latents = np.random.multivariate_normal(muX, cov=covX, size=numSamples)
    observed = np.array(
        [x.reshape((2, 1)) + 12. + np.linalg.cholesky(covY) @ np.random.normal(size=2).reshape((2, 1)) for x in
         latents]).squeeze(2)
    postCovX = np.linalg.inv(np.linalg.inv(covY) + np.linalg.inv(covX))
    postMuX = [postCovX @ (
            np.linalg.inv(covY) @ (y - 12.).reshape((2, 1)) + np.linalg.inv(covX) @ np.array(muX).reshape((2, 1)))
               for y in observed]
    posteriors = np.array([postMuX[i] + np.linalg.cholesky(postCovX) @ np.random.normal(size=2).reshape((2, 1)) for i in
                           range(numSamples)]).squeeze(2)

    # Visualise data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(latents[:, 0], latents[:, 1], alpha=0.6, label="Latent Data")
    # ax.scatter(observed[:, 0], observed[:, 1], alpha=0.6, label="Observed Data")
    ax.scatter(posteriors[:, 0], posteriors[:, 1], alpha=0.3, label="Posterior Data")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Data Visualisation"
    ax.set_title(strtitle)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend()
    plt.show()

    # Turn to Tensor
    latents = torch.from_numpy(latents)
    observed = torch.from_numpy(observed)
    posteriors = torch.from_numpy(posteriors)
    trial_data = latents

    ts = torch.linspace(start=1e-3, end=1.,
                        steps=N)  # Instability as epsilon gets small (best epsilon varies is data-driven)

    sampless = []
    sampless.append(latents.numpy())

    for t in ts:
        epsts = torch.randn_like(trial_data)
        xts = torch.exp(-0.5 * t) * trial_data + torch.sqrt((1. - torch.exp(-t))) * epsts
        sampless.append(xts.numpy())

    reversed_sampless = []
    x = torch.randn_like(trial_data)
    reversed_sampless.append(x.numpy())

    timesteps = torch.linspace(1., end=1e-3, steps=N)

    for i in tqdm((range(N))):
        dt = 1. / N
        t = timesteps[i]
        z = torch.randn_like(trial_data)
        latent_score = - (x - torch.exp(-0.5 * t) * trial_data) / (1. - torch.exp(-t))
        x0_est = trial_data  # torch.exp(0.5 * t) * x - torch.sqrt((torch.exp(t) - 1.)) * latent_score
        observed_score = -torch.cat([torch.from_numpy(
            np.linalg.inv(covY + ((torch.exp(t) - 1.) * 0.01 * np.eye(trial_data.shape[1])).numpy())) @ (
                                             observed[s].reshape((2, 1)) - x0_est[s].reshape(
                                         (2, 1)) - 12. * torch.ones_like(x0_est[s]).reshape(
                                         (2, 1))) for s in range(numSamples)], dim=-1).T  # grad log p(y0|xt)
        g = latent_score + observed_score
        x = x + (0.5 * x + g) * dt + np.sqrt(dt) * z  # Predictor
        # TODO: Corrector!
        reversed_sampless.append(x.numpy())

    generated_samples = (reversed_sampless[-1])

    # Visualise wrt to posteriors
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(posteriors[:, 0], posteriors[:, 1], alpha=0.6, label="Original Posteriors")
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend()
    plt.show()

    # Visualise wrt latents
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(latents[:, 0], latents[:, 1], alpha=0.6, label="Original Latents")
    ax.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.6, label="Generated Samples")
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    strtitle = "Scatter Plot of Final Reverse Samples"
    ax.set_title(strtitle)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.legend()
    plt.show()

    plot_final_diffusion_marginals(posteriors.numpy(), generated_samples, timeDim=2)
    plot_final_diffusion_marginals(latents.numpy(), generated_samples, timeDim=2)
