

import matplotlib.pyplot as plt
from configs import project_config
import numpy as np
import scipy
import torch
from tqdm import tqdm
import os
from utils.data_processing import init_experiment
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching  import \
    ConditionalMarkovianTSPostMeanScoreMatching

def true_cond_mean(config, prev_path):
    if "fOU" in config.data_path:
        return (-config.mean_rev * (prev_path.squeeze(-1)-config.mean))
    else:
        return (config.mean_rev * torch.sin(prev_path.squeeze(-1)))

# Generate value of path at time "t" by running reverse diffusion
def single_time_sampling(config, data_shape,  diff_time_space, diffusion, feature, scoreModel, device, prev_path, es, ts_step):
    x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
    scores = []
    exp_scores = []
    revSDE_paths = []
    assert (0 <= es <= 20)
    for diff_index in tqdm(range(config.max_diff_steps)):
        if diff_index <= config.max_diff_steps - es:

            tau = diff_time_space[diff_index] * torch.ones((data_shape[0],)).to(device)
            try:
                scoreModel.eval()
                with torch.no_grad():
                    tau = tau * torch.ones((x.shape[0],)).to(device)
                    predicted_score = scoreModel.forward(x, conditioner=feature, times=tau)
            except TypeError as e:
                scoreModel.eval()
                with torch.no_grad():
                    tau = tau * torch.ones((x.shape[0],)).to(device)
                    eff_times = diffusion.get_eff_times(diff_times=tau)
                    eff_times = eff_times.reshape(x.shape)
                    predicted_score = scoreModel.forward(x, conditioner=feature, times=tau, eff_times=eff_times)

            score, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=x,
                                                                                  predicted_score=predicted_score,
                                                                                  diff_index=torch.Tensor(
                                                                                      [int(diff_index)]).to(device),
                                                                                  max_diff_steps=config.max_diff_steps)
            if len(score.shape) == 3 and score.shape[-1] == 1:
                score = score.squeeze(-1)
            diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=tau))).T.to(device)
            diffusion_var = 1. - diffusion_mean2
            exp_slope = -(1 / ((diffusion_var + diffusion_mean2 * ts_step))[0])
            exp_const = torch.sqrt(diffusion_mean2) * (ts_step) * true_cond_mean(config, prev_path)
            exp_score = exp_slope * (x.squeeze(-1) - exp_const)
            if len(exp_score) == 3 and exp_score.shape[0] == 1:
                exp_score = exp_score.squeeze(-1)
            # Store the score, the expected score, and the revSDE paths
            scores.append(score)
            exp_scores.append(exp_score)
            if len(x.shape) == 3 and x.shape[-1] == 1:
                revSDE_paths.append(x.squeeze(-1))
            else:
                assert (x.shape == (data_shape[0], 1))
                revSDE_paths.append(x)
            z = torch.randn_like(drift)
            x = drift + diffParam * z
        else:
            if len(exp_score) == 3 and exp_score.shape[0] == 1:
                exp_score = exp_score.squeeze(-1)
            scores.append(score)
            exp_scores.append(exp_score)
            if len(x.shape) == 3 and x.shape[-1] == 1:
                revSDE_paths.append(x.squeeze(-1))
            else:
                assert (x.shape == (data_shape[0], 1))
                revSDE_paths.append(x)
    scores = torch.flip(torch.concat(scores, dim=-1).cpu(), dims=[1])
    exp_scores = torch.flip(torch.concat(exp_scores, dim=-1).cpu(), dims=[1])
    revSDE_paths = torch.flip(torch.concat(revSDE_paths, dim=-1).cpu(), dims=[1])
    return x, scores, exp_scores, revSDE_paths
# In[5]:

# Generate sample paths from [0, ts_length]
def run_whole_ts_recursive_diffusion(config, ts_length, initial_feature_input, diffusion, scoreModel, device,
                                     diff_time_scale, data_shape, es, ts_step):
    stored_scores = []
    stored_expscores = []
    stored_revSDE_paths = []
    prev_paths = []
    cumsamples = initial_feature_input
    for t in (range(ts_length)):
        prev_paths.append(cumsamples.cpu())
        print("Sampling at real time {}\n".format(t + 1))
        scoreModel.eval()
        new_samples, scores, exp_scores, revSDE_paths = single_time_sampling(config=config, data_shape=data_shape,
                                                                             diff_time_space=diff_time_scale,
                                                                             diffusion=diffusion, scoreModel=scoreModel,
                                                                             device=device, feature=cumsamples,
                                                                             prev_path=cumsamples, es=es,ts_step=ts_step)
        stored_scores.append(scores.unsqueeze(1))
        stored_expscores.append(exp_scores.unsqueeze(1))
        stored_revSDE_paths.append(revSDE_paths.unsqueeze(1))
        cumsamples = cumsamples + new_samples
    stored_scores = torch.concat(stored_scores, dim=1)
    # assert(stored_scores.shape == (data_shape[0], T, config.max_diff_steps))
    stored_expscores = torch.concat(stored_expscores, dim=1)
    # assert(stored_expscores.shape == (data_shape[0], T, config.max_diff_steps))
    stored_revSDE_paths = torch.concat(stored_revSDE_paths, dim=1)
    # assert(stored_revSDE_paths.shape == (data_shape[0], T, config.max_diff_steps))
    prev_paths = torch.concat(prev_paths, dim=1).squeeze(-1)
    return stored_scores.cpu(), stored_expscores.cpu(), stored_revSDE_paths.cpu(), prev_paths.cpu()


# Build drift estimator
def build_drift_estimator(diffusion, ts_step, diff_time_space, score_evals, exp_scores, Xtaus):
    eff_times = diffusion.get_eff_times(torch.Tensor(diff_time_space)).cpu()#.numpy()
    beta_2_taus = torch.exp(-eff_times)
    sigma_taus = 1. - beta_2_taus
    # Compute the part of the score independent of data mean
    c1 = (sigma_taus + beta_2_taus * ts_step) * torch.exp(torch.Tensor([0.5]) * eff_times)  # * 1/beta_tau
    c2 = torch.exp(torch.Tensor([0.5]) * eff_times)  # 1/beta_tau

    drift_est = c1 * score_evals + (c2.reshape(1, 1, -1)) * Xtaus
    drift_est /= ts_step

    exp_drifts = c1 * exp_scores + (c2.reshape(1, 1, -1)) * Xtaus
    exp_drifts /= ts_step
    return drift_est.cpu(), exp_drifts.cpu()
def TSPM_drift_eval():
    from configs.RecursiveVPSDE.recursive_Markovian_PostMeanScaledScore_fSin_T256_H05_tl_5data import get_config as get_config_postmean
    config_postmean = get_config_postmean()
    init_experiment(config=config_postmean)

    num_simulated_paths = 50
    data_shape = (num_simulated_paths, 1, 1)

    if config_postmean.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")

    revDiff_time_scale = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
                                     steps=config_postmean.max_diff_steps).to(device)
    diffusion = VPSDEDiffusion(beta_max=config_postmean.beta_max, beta_min=config_postmean.beta_min)


    ts_step = 1 / config_postmean.ts_length

    Nepoch = 960#config_postmean.max_epochs[0]
    es = 0
    assert (config_postmean.max_diff_steps == 10000)
    if "fOU" in config_postmean.data_path:
        save_path = (project_config.ROOT_DIR + f"experiments/results/TSPMS_mkv_ES{es}_DriftEvalExp_{Nepoch}Nep_{config_postmean.loss_factor}LFactor_{config_postmean.mean}Mean_{config_postmean.max_diff_steps}DiffSteps").replace(".", "")
    elif "fSin" in config_postmean.data_path:
        save_path = (project_config.ROOT_DIR + f"experiments/results/TSPMS_mkv_ES{es}_fSin_DriftEvalExp_{Nepoch}Nep_{config_postmean.loss_factor}LFactor_{config_postmean.mean_rev}MeanRev_{config_postmean.max_diff_steps}DiffSteps").replace(".", "")

    # Fix the number of training epochs and training loss objective loss
    PM = ConditionalMarkovianTSPostMeanScoreMatching(*config_postmean.model_parameters).to(device)
    PM.load_state_dict(torch.load(config_postmean.scoreNet_trained_path + "_NEp" + str(Nepoch)))

    print(Nepoch, config_postmean.data_path, es, config_postmean.scoreNet_trained_path)
    # Fix the number of real times to run diffusion
    eval_ts_length = int(1.3*config_postmean.ts_length)
    # Experiment for score model with fixed (Nepochs, loss scaling, drift eval time, Npaths simulated)
    initial_feature_input = torch.zeros(data_shape).to(device)
    postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
        ts_length=eval_ts_length, config=config_postmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
        scoreModel=PM, device=device, diff_time_scale=revDiff_time_scale, data_shape=data_shape, es=es, ts_step=ts_step)

    # Output shape is (NumPaths, NumRealTimes, NumDiffSteps)
    torch.save(postMean_scores, save_path + "_scores")
    torch.save(postMean_revSDEpaths, save_path + "_paths")


if __name__ == "__main__":
    TSPM_drift_eval()