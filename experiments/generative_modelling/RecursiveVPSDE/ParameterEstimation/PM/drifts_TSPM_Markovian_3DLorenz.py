#!/usr/bin/env python
# coding: utf-8

# In[21]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from configs.RecursiveVPSDE.Markovian_3DLorenz.recursive_Markovian_PostMeanScore_3DLorenz_T256_H05_tl_5data import \
    get_config as get_config
from tqdm import tqdm

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import \
    ConditionalMarkovianTSScoreMatching

# In[22]:


config = get_config()
assert (config.max_diff_steps == 10000 and config.beta_min == 0.)
print("Beta Min : ", config.beta_min)
if config.has_cuda:
    device = int(os.environ["LOCAL_RANK"])
else:
    print("Using CPU\n")
    device = torch.device("cpu")

diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

max_diff_steps = config.max_diff_steps
sample_eps = config.sample_eps
ts_step = 1 / config.ts_length

Nepoch = 960  # config.max_epochs[0]
# Fix the number of training epochs and training loss objective loss
if "PM" in config.scoreNet_trained_path:
    PM = ConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters).to(device)
else:
    PM = ConditionalMarkovianTSScoreMatching(*config.model_parameters).to(device)
#PM.load_state_dict(
#    torch.load(config.scoreNet_trained_path.replace("trained_models", "snapshots"), map_location=torch.device('cpu'))[
#        "MODEL_STATE"])  # + "_NEp" + str(Nepoch)))
PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
# In[23]:


Xshape = config.ts_length
num_taus = 50
num_diff_times = 10000
assert (num_diff_times * num_taus == 500000)
Ndiff_discretisation = config.max_diff_steps
diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                 steps=Ndiff_discretisation).to(device)

# Create a uniform grid
x = np.linspace(-20, 20, 10)
y = np.linspace(-30, 30, 10)
z = np.linspace(0, 50, 10)

# Create a 3D mesh grid
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
D = 3
# Combine into a single array of shape (n^3, 3) if needed
Xs = torch.Tensor(np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)[np.newaxis, :, :]).to(device)
Xshape = Xs.shape[1]
B = Xs.shape[1]
conditioner = Xs.permute(1, 0, 2)
mu_hats = np.zeros((Xshape, D, num_diff_times, num_taus))  # Xvalues, TS Dimension, DiffTimes, Ztaus
PM.eval()
for k in tqdm(range(num_taus)):
    difftime_idx = num_diff_times - 1
    Z_taus = diffusion.prior_sampling(shape=(Xshape, 1, D)).to(device)
    while difftime_idx >= 0:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(B)]).reshape(B * 1, 1, -1).squeeze(-1).squeeze(-1).squeeze(-1).to(
            device)
        eff_times = torch.cat([diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1)] * D,
                              dim=2).to(device)
        if k == 0 and difftime_idx < 100:
            print(d)
        with torch.no_grad():
            if "PM" in config.scoreNet_trained_path:
                print(Z_taus.shape, diff_times.shape, conditioner.shape, eff_times.shape)
                predicted_score = PM.forward(inputs=Z_taus, times=diff_times, conditioner=conditioner,
                                             eff_times=eff_times)
            else:
                predicted_score = PM.forward(inputs=Z_taus, times=diff_times, conditioner=conditioner)
            scores, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=Z_taus,
                                                                                   predicted_score=predicted_score,
                                                                                   diff_index=torch.Tensor(
                                                                                       [int((
                                                                                               num_diff_times - 1 - difftime_idx))]).to(
                                                                                       device),
                                                                                   max_diff_steps=Ndiff_discretisation)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma_taus = torch.pow(1. - torch.pow(beta_taus, 2), 0.5).to(device)
        for i in range(Xshape):
            Zts = Z_taus[i, :, :]
            Ss = scores[i, :, :]
            mu_hat = Zts / (ts_step * beta_taus) + (
                    (torch.pow(sigma_taus, 2) + (torch.pow(beta_taus, 2) * ts_step)) / (ts_step * beta_taus)) * Ss
            mu_hats[i, :, difftime_idx, k] = mu_hat[0, :].cpu().detach().numpy()
        z = torch.randn_like(drift).to(device)
        Z_taus = drift + diffParam * z
        difftime_idx -= 1

numpy_Xs = Xs.squeeze(0).cpu().detach().numpy()
if "PMS" in config.scoreNet_trained_path:
    type = "PMS"
elif "PM" in config.scoreNet_trained_path:
    type = "PM"
else:
    type = "Standard"
print(type)

es = 0

if "fOU" in config.data_path:
    save_path = \
        (
                project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_ES{es}_DriftEvalExp_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean}Mean_{config.max_diff_steps}DiffSteps").replace(
            ".", "")
elif "3DLorenz" in config.data_path:
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_ES{es}_3DLorenz_DriftEvalExp_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean_rev}MeanRev_{config.max_diff_steps}DiffSteps").replace(
        ".", "")

np.save(save_path + "_muhats.npy", mu_hats)
np.save(save_path + "_numpyXs.npy", numpy_Xs)