#!/usr/bin/env python
# coding: utf-8

# In[21]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

from configs.RecursiveVPSDE.recursive_Markovian_fSinWithPosition_T256_H05_tl_5data import get_config as get_config

from configs import project_config
import numpy as np
import torch
import os
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching  import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import ConditionalMarkovianTSScoreMatching
from tqdm import tqdm

import matplotlib.pyplot as plt


# In[22]:


config = get_config()
print(config.beta_min)
if config.has_cuda:
    device = int(os.environ["LOCAL_RANK"])
else:
    print("Using CPU\n")
    device = torch.device("cpu")

diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

max_diff_steps = config.max_diff_steps
sample_eps = config.sample_eps
ts_step = 1 / config.ts_length

Nepoch = 960#config.max_epochs[0]
# Fix the number of training epochs and training loss objective loss
if "PM" in config.scoreNet_trained_path:
    PM = ConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters).to(device)
else:
    PM = ConditionalMarkovianTSScoreMatching(*config.model_parameters).to(device)
PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))


# In[23]:


Xshape = config.ts_length
num_taus = 20
num_diff_times = 10000
Ndiff_discretisation = num_diff_times  #config.max_diff_steps
diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                 steps=Ndiff_discretisation).to(device)
mu_hats_mean = np.zeros((Xshape, num_taus))
mu_hats_std = np.zeros((Xshape, num_taus))

Xs = torch.linspace(-2, 2, steps=Xshape).unsqueeze(-1).unsqueeze(-1).permute(1,0,2)
conditioner = torch.stack([Xs for _ in range(1)], dim=0).reshape(Xshape*1, 1, -1)
B, T = Xshape, 1
mu_hats = np.zeros((Xshape, num_diff_times, num_taus))  # Xvalues, DiffTimes, Ztaus
PM.eval()
for k in tqdm(range(num_taus)):
    difftime_idx = num_diff_times - 1
    Z_taus = diffusion.prior_sampling(shape=(Xshape, 1, 1))
    while difftime_idx >= 0:
        d = diffusion_times[Ndiff_discretisation-(num_diff_times - 1 - difftime_idx)-1]
        diff_times = torch.stack([d for _ in range(B)]).reshape(B*T, 1, -1).squeeze(-1).squeeze(-1).squeeze(-1)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1)
        if k == 0 and difftime_idx < 100:
            print(d)
        with torch.no_grad():
            if "PM" in config.scoreNet_trained_path:
                predicted_score = PM.forward(inputs=Z_taus, times=diff_times, conditioner=conditioner, eff_times=eff_times)
            else:
                predicted_score = PM.forward(inputs=Z_taus, times=diff_times, conditioner=conditioner)
            scores, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=Z_taus,
                                                                                      predicted_score=predicted_score,
                                                                                      diff_index=torch.Tensor(
                                                                                          [int((num_diff_times - 1 - difftime_idx))]).to(device),
                                                                                      max_diff_steps=Ndiff_discretisation)
        #assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0,0,0])
        sigma_taus = torch.pow(1. - torch.pow(beta_taus, 2), 0.5)
        for i in range(Xshape):
            Zts = Z_taus[i, :,:]
            Ss = scores[i,:,:]
            mu_hat = Zts/(ts_step*beta_taus) + ((torch.pow(sigma_taus, 2)+(torch.pow(beta_taus,2)*ts_step))/(ts_step*beta_taus))*Ss
            mu_hats[i, difftime_idx, k] = mu_hat[0,0]
        z = torch.randn_like(drift)
        Z_taus = drift + diffParam * z
        difftime_idx -= 1


def plot_drift_estimator(mean, stds, numpy_Xs, type, toSave:bool = True):
    fig, ax = plt.subplots(figsize=(14,9))
    rmse = np.power(np.mean(np.power(np.sin(numpy_Xs) - mean, 2)), 0.5)
    ax.scatter(numpy_Xs, np.sin(numpy_Xs), color="red", label="True Drift")

    ax.errorbar(numpy_Xs, mean, fmt="o",yerr=2*stds, label="Drift Estimator with 2 Std")
    ax.set_title(rf"RMSE {round(rmse,3)} of estimator $\bar{{\mu}}(x)$", fontsize=20)
    ax.tick_params("x",labelsize=18)
    ax.tick_params("y",labelsize=18)
    ax.set_xlabel("State $x$", fontsize=18)
    ax.set_ylabel("Drift Value", fontsize=18)
    ax.legend(loc="lower right", fontsize=18)
    if toSave:
        plt.savefig(f"/Users/marcos/Library/CloudStorage/OneDrive-ImperialCollegeLondon/StatML_CDT/Year2/DiffusionModelPresentationImages/fSin_{type}.png")
    plt.show()
    plt.close()


# In[25]:


print(Xs.shape)
try:
    numpy_Xs = Xs.numpy().flatten()
    #numpy_Xs = Xs[:,int(0.4*Xshape):int(0.6*Xshape)+1,:][:,:-1,:].numpy().flatten()
    #mu_hats = mu_hats[int(0.4*Xshape):int(0.6*Xshape),:,:]
except (IndexError, AttributeError) as e:
    print(e)
    assert (numpy_Xs.shape[1] == Xshape)
    pass

if "PMS" in config.scoreNet_trained_path:
    type="PMS"
elif "PM" in config.scoreNet_trained_path:
    type = "PM"
else:
    type = "Standard"
print(type)

es = 0

if "fOU" in config.data_path:
    save_path = \
        (
                    project_config.ROOT_DIR + f"experiments/results/TS_mkv_ES{es}_DriftEvalExp_{Nepoch}Nep_{0}LFactor_{config.mean}Mean_{config.max_diff_steps}DiffSteps").replace(
            ".", "")
elif "fSin" in config.data_path:
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/TS_mkv_ES{es}_fSin_DriftEvalExp_{Nepoch}Nep_{0}LFactor_{config.mean_rev}MeanRev_{config.max_diff_steps}DiffSteps").replace(
        ".", "")

np.save(save_path + "_muhats.npy", mu_hats)
np.save(save_path + "_numpyXs.npy", numpy_Xs)

raise RuntimeError

for j in range(0, num_diff_times, 10):
    mhats = mu_hats[:,j,:]
    mhats = mhats.reshape(mhats.shape[0], np.prod(mhats.shape[1:]))
    mean = mhats.mean(axis=-1)
    stds = mhats.std(axis=-1)
    plot_drift_estimator(mean, stds, numpy_Xs, type=type, toSave=False)


# In[34]:


mean = np.array([mu_hats[i,10:200,:].flatten().mean(axis=-1) for i in range(Xshape)])
stds = np.array([mu_hats[i,10:200,:].flatten().std(axis=-1) for i in range(Xshape)])
plot_drift_estimator(mean, stds, numpy_Xs, type=type, toSave=False)


# In[ ]:





# In[21]:





# In[ ]:





# In[7]:



