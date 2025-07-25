
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from configs.RecursiveVPSDE.Markovian_fSin.recursive_Markovian_PostMeanScore_fSin_T256_H05_tl_5data import get_config as get_config
from tqdm import tqdm

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import \
    ConditionalMarkovianTSScoreMatching

# In[22]:


config = get_config()
print("Beta Min : ", config.beta_min)
if config.has_cuda:
    device = int(os.environ["LOCAL_RANK"])
else:
    print("Using CPU\n")
    device = torch.device("cpu")

diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

max_diff_steps = config.max_diff_steps
sample_eps = config.sample_eps
ts_step = config.deltaT

Nepoch = 960  # config.max_epochs[0]
# Fix the number of training epochs and training loss objective loss
if "PM" in config.scoreNet_trained_path:
    PM = ConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters).to(device)
else:
    PM = ConditionalMarkovianTSScoreMatching(*config.model_parameters).to(device)
PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
print(config.scoreNet_trained_path)

# In[23]:


Xshape = config.ts_length
num_taus = 200
num_diff_times = config.max_diff_steps
Ndiff_discretisation = config.max_diff_steps
diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                 steps=Ndiff_discretisation).to(device)
mu_hats_mean = np.zeros((Xshape, num_taus))
mu_hats_std = np.zeros((Xshape, num_taus))

Xs = torch.linspace(-3, 3, steps=Xshape).unsqueeze(-1).unsqueeze(-1).permute(1, 0, 2).to(device)
conditioner = torch.stack([Xs for _ in range(1)], dim=0).reshape(Xshape * 1, 1, -1)
B, T = Xshape, 1
final_vec_mu_hats = np.zeros((Xshape, num_diff_times, num_taus))  # Xvalues, DiffTimes, Ztaus
PM.eval()
vec_Z_taus = diffusion.prior_sampling(shape=(Xshape*num_taus, 1, 1)).to(device)
difftime_idx = num_diff_times - 1
ts = []
es = 1
while difftime_idx >= num_diff_times - es:
    d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
    # I (will) have a RV for each x (there are B of them) and hence need a diffusion time for each one
    diff_times = torch.stack([d for _ in range(B)]).reshape(B * T).to(device)
    eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
    vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus*Xshape).to(device)
    vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus*Xshape, 1, 1).to(device)
    vec_conditioner = torch.stack([conditioner for _ in range(num_taus)], dim=0).reshape(num_taus*Xshape, 1, 1)

    with torch.no_grad():
        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                         eff_times=vec_eff_times)
        else:
            vec_predicted_score = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner)
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                               predicted_score=vec_predicted_score,
                                                                               diff_index=torch.Tensor(
                                                                                   [int((
                                                                                           num_diff_times - 1 - difftime_idx))]).to(
                                                                                   device),
                                                                               max_diff_steps=Ndiff_discretisation)
    # assert np.allclose((scores- predicted_score).detach(), 0)
    beta_taus = torch.exp(-0.5 * eff_times[0,0,0]).to(device)
    sigma_taus = torch.pow(1. - torch.pow(beta_taus, 2), 0.5).to(device)
    final_mu_hats = (vec_Z_taus/(ts_step * beta_taus)) + ( (
                (torch.pow(sigma_taus, 2) + (torch.pow(beta_taus, 2) * ts_step)) / (ts_step * beta_taus)) * vec_scores)
    #print(vec_Z_taus.shape, vec_scores.shape)
    final_vec_mu_hats[:, difftime_idx, :] = final_mu_hats.reshape((num_taus, Xshape)).T.cpu().numpy()
    vec_z = torch.randn_like(vec_drift).to(device)
    vec_Z_taus = vec_drift + vec_diffParam * vec_z
    difftime_idx -= 1

# In
type = "PM"
assert (type in config.scoreNet_trained_path)
print(type)

save_path = (
            project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_fSin_DriftEvalExp_{Nepoch}Nep_{config.mean_rev}MeanRev_{config.max_diff_steps}DiffSteps_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
        ".", "")
np.save(save_path + "_muhats.npy", final_vec_mu_hats[:, -es:, :])