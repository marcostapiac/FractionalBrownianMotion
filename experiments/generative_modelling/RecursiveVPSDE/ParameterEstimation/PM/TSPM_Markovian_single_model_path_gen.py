from configs import project_config
import torch
from tqdm import tqdm
import os
from utils.data_processing import init_experiment
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
import os

import torch
from tqdm import tqdm

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from utils.data_processing import init_experiment


def true_cond_mean(config, prev_path):
    if "fOU" in config.data_path:
        return (-config.mean_rev * (prev_path.squeeze(-1) - config.mean))
    else:
        return (config.mean_rev * torch.sin(prev_path.squeeze(-1)))


# Generate value of path at time "t" by running reverse diffusion
def single_time_sampling(config, data_shape, diff_time_space, diffusion, feature, scoreModel, device, prev_path, es,
                         ts_step):
    x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
    scores = []
    exp_scores = []
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
            z = torch.randn_like(drift)
            x = drift + diffParam * z
        else:
            if len(exp_score) == 3 and exp_score.shape[0] == 1:
                exp_score = exp_score.squeeze(-1)
            scores.append(score)
            exp_scores.append(exp_score)
    scores = torch.flip(torch.concat(scores, dim=-1).cpu(), dims=[1])
    exp_scores = torch.flip(torch.concat(exp_scores, dim=-1).cpu(), dims=[1])
    return x, scores, exp_scores


# In[5]:

# Generate sample paths from [0, ts_length]
def run_whole_ts_recursive_diffusion(config, ts_length, initial_feature_input, diffusion, scoreModel, device,
                                     diff_time_scale, data_shape, es, ts_step):
    stored_scores = []
    stored_expscores = []
    paths = []
    cumsamples = initial_feature_input
    for t in (range(ts_length)):
        paths.append(cumsamples.cpu())
        print("Sampling at real time {}\n".format(t + 1))
        scoreModel.eval()
        new_samples, scores, exp_scores = single_time_sampling(config=config, data_shape=data_shape,
                                                               diff_time_space=diff_time_scale,
                                                               diffusion=diffusion, scoreModel=scoreModel,
                                                               device=device, feature=cumsamples,
                                                               prev_path=cumsamples, es=es, ts_step=ts_step)
        stored_scores.append(scores.unsqueeze(1))
        stored_expscores.append(exp_scores.unsqueeze(1))
        cumsamples = cumsamples + new_samples
    paths.append(cumsamples.cpu())
    stored_scores = torch.concat(stored_scores, dim=1)
    # assert(stored_scores.shape == (data_shape[0], T, config.max_diff_steps))
    stored_expscores = torch.concat(stored_expscores, dim=1)
    # assert(stored_expscores.shape == (data_shape[0], T, config.max_diff_steps))
    paths = torch.concat(paths, dim=1).squeeze(-1)
    return stored_scores.cpu(), stored_expscores.cpu(), paths.cpu()


# Build drift estimator
def build_drift_estimator(diffusion, ts_step, diff_time_space, score_evals, exp_scores, Xtaus):
    eff_times = diffusion.get_eff_times(torch.Tensor(diff_time_space)).cpu()  # .numpy()
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
    from configs.RecursiveVPSDE.recursive_Markovian_PostMeanScore_fSin_T256_H05_tl_5data import get_config
    config = get_config()
    init_experiment(config=config)

    num_simulated_paths = 10000
    data_shape = (num_simulated_paths, 1, 1)

    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")

    revDiff_time_scale = torch.linspace(start=config.end_diff_time, end=config.sample_eps,
                                        steps=config.max_diff_steps).to(device)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    ts_step = 1 / config.ts_length

    Nepoch = 960
    for es in [0, 10, 100, 200]:
        assert (config.max_diff_steps == 10000)
        if "fOU" in config.data_path:
            save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_ES{es}_PathGen_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean}Mean_{config.max_diff_steps}DiffSteps").replace(
                ".", "")
        elif "fSin" in config.data_path:
            save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_mkv_ES{es}_fSin_PathGen_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean_rev}MeanRev_{config.max_diff_steps}DiffSteps").replace(
                ".", "")

        # Fix the number of training epochs and training loss objective loss
        PM = ConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters).to(device)
        PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))

        print(Nepoch, config.data_path, es, config.scoreNet_trained_path)
        # Fix the number of real times to run diffusion
        eval_ts_length = int(1.3 * config.ts_length)
        # Experiment for score model with fixed (Nepochs, loss scaling, drift eval time, Npaths simulated)
        initial_feature_input = torch.zeros(data_shape).to(device)
        postMean_scores, postMean_expscores, paths = run_whole_ts_recursive_diffusion(
            ts_length=eval_ts_length, config=config, initial_feature_input=initial_feature_input, diffusion=diffusion,
            scoreModel=PM, device=device, diff_time_scale=revDiff_time_scale, data_shape=data_shape, es=es,
            ts_step=ts_step)

        torch.save(postMean_scores, save_path + "_scores")
        torch.save(postMean_expscores, save_path + "_exp_scores")
        torch.save(paths, save_path + "_paths")


if __name__ == "__main__":
    TSPM_drift_eval()
