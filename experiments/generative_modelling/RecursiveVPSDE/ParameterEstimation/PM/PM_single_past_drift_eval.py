import matplotlib
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
N = 2
data_shape = (N, 1, 1)
device = "cpu"

diff_time_space = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
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

true_paths = np.load(config_postmean.data_path, allow_pickle=True)[:N,:]
true_paths = true_paths.reshape((true_paths.shape[0], true_paths.shape[1], 1))
true_paths = np.concatenate([np.zeros((true_paths.shape[0],1,true_paths.shape[-1])), true_paths], axis=1)
start_time_idx = 200
end_time_idx = 256
end_diff_idx = 9100
drifts = np.zeros(shape=(end_time_idx-start_time_idx, max_diff_steps-end_diff_idx))
path_values = []
PM_960.eval()
with torch.no_grad():
    tensor_true_paths = torch.Tensor(true_paths).to(device)
    output, (hn, cn) = (PM_960.rnn(tensor_true_paths, None))
    features = output[:, :-1, :]
    # Fix a single past feature for a single time
    del tensor_true_paths, output

for i in tqdm(range(start_time_idx,end_time_idx)):
    pathidx = np.random.randint(low=0, high=true_paths.shape[0])
    path_values.append(true_paths[[pathidx], [i],:].flatten()[0])
    fixed_feature = torch.tile(features[[pathidx], [i], :].unsqueeze(0),dims=(N, 1,1))
    # Now run reverse-diffusion
    with torch.no_grad():
        diff_time_space = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
                                        steps=config_postmean.max_diff_steps).to(device)
        x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
        for diff_index in (range(0, config_postmean.max_diff_steps - end_diff_idx)):
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
            #if len(score.shape) == 3 and score.shape[-1] == 1:
            #    score = score.squeeze(-1)
            # Evaluate the drifts means
            diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=tau[0]))).T.detach().cpu().squeeze()
            diffusion_mean = torch.atleast_2d(torch.exp(-0.5*diffusion.get_eff_times(diff_times=tau[0]))).T.detach().cpu().squeeze()
            diffusion_var = 1. - diffusion_mean2
            drift_est = ((diffusion_var + diffusion_mean2*ts_step)/(diffusion_mean*ts_step))*score.detach().cpu() + x.detach().cpu()/(diffusion_mean*ts_step)
            drifts[i-start_time_idx,diff_index] = torch.mean(drift_est).flatten().detach().cpu()[0]
            z = torch.randn_like(drift)
            x = drift + diffParam * z
assert(len(path_values) == drifts.shape[0])
sorted_idxs = np.argsort(path_values)
path_values = np.array(path_values)[sorted_idxs]
drifts = drifts[sorted_idxs, :]
# Separate the pairs back into two arrays
ax = plt.axes(projection='3d')
x, y = np.meshgrid(diff_time_space[:config_postmean.max_diff_steps-end_diff_idx], path_values) # x is the columns, y is the rows
ax.scatter3D(x, y, drifts, cmap='viridis', label="Estimated")
ax.scatter3D(x, y,  -config_postmean.mean_rev*path_values.reshape((path_values.shape[0],1))*np.ones((drifts.shape[0], drifts.shape[1])), label="Expected")
plt.legend()
ax.set_xlabel('Diffusion Time', rotation=150)
ax.set_ylabel('Path/State Value',  rotation=150)
plt.show()
plt.close()