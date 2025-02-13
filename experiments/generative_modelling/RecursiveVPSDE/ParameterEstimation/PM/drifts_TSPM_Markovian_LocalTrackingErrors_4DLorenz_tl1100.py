#!/usr/bin/env python
# coding: utf-8

# In[21]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import os
import numpy as np
import torch
from configs.RecursiveVPSDE.Markovian_4DLorenz.recursive_Markovian_PostMeanScore_4DLorenz_T256_H05_tl_1100data import \
    get_config as get_config
from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from tqdm import tqdm
# In[22]:


config = get_config()
assert ("4DLnz" in config.data_path)

print("Beta Min : ", config.beta_min)
if config.has_cuda:
    device = int(os.environ["LOCAL_RANK"])
else:
    print("Using CPU\n")
    device = torch.device("cpu")

diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

Nepoch = 960  # config.max_epochs[0]
num_diff_times = 1000
# Fix the number of training epochs and training loss objective loss
PM = ConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters).to(device)
PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))

num_paths = 10
num_time_steps = 50
deltaT = 1. / 256
initial_state = np.repeat(np.array(config.initState)[np.newaxis, np.newaxis, :], num_paths, axis=0)
assert (initial_state.shape == (num_paths, 1, config.ndims))

true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
local_states = np.zeros(shape=(num_diff_times, num_paths, 1 + num_time_steps, config.ndims))
#global_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

# Initialise the "true paths"
true_states[:, [0], :] = initial_state
# Initialise the "local drift approximation paths"
local_states[:,:, [0], :] = np.repeat(initial_state[np.newaxis,:], num_diff_times, axis=0)
# Initialise the "global score-based drift paths"
#global_states[:, [0], :] = np.repeat(initial_state[np.newaxis,:], num_diff_times, axis=0)


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = np.zeros((num_paths, config.ndims))
    for i in range(config.ndims):
        drift_X[:,i] = (prev[:, (i + 1) % config.ndims] - prev[:, i - 2]) * prev[:, i - 1] - prev[:,
                                                                                           i] + config.forcing_const
    return drift_X[:, np.newaxis, :]


def local_score_based_drift(score_model, num_diff_times, diffusion, num_paths, prev, ts_step, config, device):
    num_taus = 500
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    conditioner = torch.Tensor(prev[:, np.newaxis, :]).to(device)  # TODO: Check this is how we condition wheen D>1
    vec_conditioner = torch.stack([conditioner for _ in range(num_taus)], dim=0).reshape(num_taus * num_paths, 1,
                                                                                         config.ndims).to(device)
    vec_Z_taus = diffusion.prior_sampling(shape=(num_taus * num_paths, 1, config.ndims)).to(device)

    mean_drifts = np.zeros((num_diff_times, num_paths, 1, config.ndims))
    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.Tensor([d]).to(device)
        eff_times = diffusion.get_eff_times(diff_times).to(device)
        vec_diff_times = torch.concat([diff_times for _ in range(num_taus * num_paths)], dim=0).to(device)
        vec_eff_times = torch.concat(
            [torch.concat([eff_times.unsqueeze(-1).unsqueeze(-1) for _ in range(num_taus * num_paths)], dim=0) for _ in
             range(config.ndims)], dim=-1).to(device)
        score_model.eval()
        with torch.no_grad():
            vec_predicted_score = score_model.forward(times=vec_diff_times, eff_times=vec_eff_times,
                                                      conditioner=vec_conditioner, inputs=vec_Z_taus)
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int(0)]).to(device),
                                                                                           max_diff_steps=Ndiff_discretisation)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times).to(device)
        sigma_taus = torch.pow(1. - torch.pow(beta_taus, 2), 0.5).to(device)
        drifts = (vec_Z_taus / (ts_step * beta_taus)) + (((torch.pow(sigma_taus, 2) + (
                torch.pow(beta_taus, 2) * ts_step)) / (
                                                                  ts_step * beta_taus)) * vec_scores)
        drifts = drifts.reshape((num_taus, num_paths, 1, config.ndims)).permute((1, 0, 2, 3))
        assert (drifts.shape == (num_paths, num_taus, 1, config.ndims))
        means = drifts.mean(dim=1)
        assert (means.shape == (num_paths, 1, config.ndims))
        mean_drifts[difftime_idx - Ndiff_discretisation + num_diff_times, :, :, :] = means.cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return mean_drifts


# Euler-Maruyama Scheme for Tracking Errors
for i in tqdm(range(1, num_time_steps+1)):
    eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)
    assert (eps.shape == (num_paths, 1, config.ndims))
    true_states[:, [i], :] = true_states[:, [i - 1], :] \
                             + true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config) * deltaT \
                             + eps
    local_mean = local_score_based_drift(score_model=PM, num_diff_times=num_diff_times, diffusion=diffusion,
                                   num_paths=num_paths, ts_step=deltaT, config=config, device=device,
                                   prev=true_states[:, i - 1, :])
    local_states[:, :, [i], :] = np.repeat(true_states[:, [i - 1], :][np.newaxis, :, :, :], num_diff_times,
                                           axis=0) + local_mean * deltaT + np.repeat(eps[np.newaxis, :, :, :],
                                                                                     num_diff_times, axis=0)
    #global_mean = score_based_drift(score_model=PM, num_diff_times=num_diff_times, diffusion=diffusion,
    #                                num_paths=num_paths, ts_step=deltaT, config=config, device=device,
    #                                prev=global_states[:, i - 1, :])

    #global_states[:, [i], :] = global_states[:, [i - 1], :] \
    #                           + global_mean * deltaT + eps


save_path = (
        project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_{config.ndims}DLorenz_DriftEvalExp_{Nepoch}Nep_tl{config.tdata_mult}data_{config.max_diff_steps}DiffSteps").replace(
    ".", "")
print(save_path)
np.save(save_path + "_true_states.npy", true_states)
np.save(save_path + "_local_states.npy", local_states)
#np.save(save_path + "_global_states.npy", global_states)
