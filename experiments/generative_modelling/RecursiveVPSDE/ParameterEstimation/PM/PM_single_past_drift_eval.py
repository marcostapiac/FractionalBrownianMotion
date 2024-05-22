
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from tqdm import tqdm

from configs.RecursiveVPSDE.recursive_PostMeanScore_fOU_T256_H07_tl_5data import get_config as get_config_postmean
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching

# In[2]:


config_postmean = get_config_postmean()

rng = np.random.default_rng()
N = 10000
data_shape = (N, 1, 1)
device = "cuda:0"

diff_time_scale = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
                                 steps=config_postmean.max_diff_steps).to(device)
real_time_scale = torch.linspace(start=1 / config_postmean.ts_length, end=1, steps=config_postmean.ts_length).to(device)
diffusion = VPSDEDiffusion(beta_max=config_postmean.beta_max, beta_min=config_postmean.beta_min)
ts_length = config_postmean.ts_length
max_diff_steps = config_postmean.max_diff_steps
sample_eps = config_postmean.sample_eps
mean_rev = config_postmean.mean_rev
ts_step = 1 / ts_length

# In[3]:


PM_960 = ConditionalLSTMTSPostMeanScoreMatching(*config_postmean.model_parameters).to(device)
PM_960.load_state_dict(torch.load(config_postmean.scoreNet_trained_path + "_NEp" + str(960)))

true_paths = np.load(config_postmean.data_path, allow_pickle=True)
true_paths = torch.Tensor(true_paths.reshape((true_paths.shape[0], true_paths.shape[1], 1))).to(device)
true_paths = torch.cat([torch.zeros((true_paths.shape[0], 1, true_paths.shape[-1])).to(device), true_paths], dim=1)
output, (hn, cn) = (PM_960.rnn(true_paths, None))
features = output[:, :-1, :]
# Fix a single past feature for a single time
fixed_feature = output[[0], [123], :]
print(fixed_feature.shape)
assert(fixed_feature.shape == (1,1,1))
fixed_feature = torch.tile(fixed_feature,dims=(N, 1,1))

# Now run reverse-diffusion
diff_time_space = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
                                 steps=config_postmean.max_diff_steps).to(device)
x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
for diff_index in tqdm(0, config_postmean.max_diff_steps - 900):
    tau = diff_time_space[diff_index] * torch.ones((data_shape[0],)).to(device)
    tau = tau * torch.ones((x.shape[0],)).to(device)
    eff_times = diffusion.get_eff_times(diff_times=tau)
    eff_times = eff_times.reshape(x.shape)
    predicted_score = PM_960.forward(x, conditioner=fixed_feature, times=tau, eff_times=eff_times)

    score, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=x,
                                                                          predicted_score=predicted_score,
                                                                          diff_index=torch.Tensor(
                                                                              [int(diff_index)]).to(device),
                                                                          max_diff_steps=config_postmean.max_diff_steps)
    if len(score.shape) == 3 and score.shape[-1] == 1:
        score = score.squeeze(-1)
    diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=tau))).T.to(device)
    diffusion_mean = torch.atleast_2d(torch.exp(-0.5*diffusion.get_eff_times(diff_times=tau))).T.to(device)
    diffusion_var = 1. - diffusion_mean2
    if diff_index == config_postmean.max_diff_steps - 900 -1:
        # Evaluate the drifts means
        drift_est = ((diffusion_var + diffusion_mean2*ts_step)/(diffusion_mean*ts_step))*score + x/(diffusion_mean*ts_step)
    z = torch.randn_like(drift)
    x = drift + diffParam * z