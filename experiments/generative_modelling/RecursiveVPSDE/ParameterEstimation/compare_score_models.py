#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from configs.RecursiveVPSDE.recursive_LSTM_PostMeanScore_fOU_T256_H05_tl_5data import get_config as get_config_postmean
from configs.RecursiveVPSDE.recursive_LSTM_fOU_T256_H05_tl_5data import get_config as get_config_score
from configs.RecursiveVPSDE.recursive_LSTM_fOU_T256_H05_tl_5data_scaled import get_config as get_config_score_scaled
from configs.RecursiveVPSDE.recursive_rPostMeanScore_fOU_T256_H07_tl_5data import get_config as get_config_rpostmean
from configs.RecursiveVPSDE.recursive_rrPostMeanScore_fOU_T256_H07_tl_5data import get_config as get_config_rrpostmean
from configs.RecursiveVPSDE.recursive_rrrPostMeanScore_fOU_T256_H07_tl_5data import get_config as get_config_rrrpostmean
from configs.RecursiveVPSDE.recursive_rrrrPostMeanScore_fOU_T256_H07_tl_5data import \
    get_config as get_config_rrrrpostmean
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSScoreMatching import \
    ConditionalLSTMTSScoreMatching

# In[2]:


config_postmean = get_config_postmean()
config_rpostmean = get_config_rpostmean()
config_rrpostmean = get_config_rrpostmean()
config_rrrpostmean = get_config_rrrpostmean()
config_rrrrpostmean = get_config_rrrrpostmean()
config_score = get_config_score()
config_score_scaled = get_config_score_scaled()

rng = np.random.default_rng()
train_epoch = 960
N = 300
data_shape = (N, 1, 1)
device = "cpu"
assert (
        config_postmean.end_diff_time == config_score.end_diff_time and config_postmean.sample_eps == config_score.sample_eps and config_postmean.max_diff_steps == config_score.max_diff_steps)
diff_time_scale = torch.linspace(start=config_postmean.end_diff_time, end=config_postmean.sample_eps,
                                 steps=config_postmean.max_diff_steps)
real_time_scale = torch.linspace(start=1 / config_postmean.ts_length, end=1, steps=config_postmean.ts_length)
assert (config_postmean.beta_max == config_score.beta_max and config_postmean.beta_min == config_score.beta_min)
diffusion = VPSDEDiffusion(beta_max=config_score.beta_max, beta_min=config_score.beta_min)
assert (config_score.ts_length == config_postmean.ts_length)
ts_length = config_postmean.ts_length
max_diff_steps = config_postmean.max_diff_steps
sample_eps = config_postmean.sample_eps
assert (config_postmean.mean_rev == config_score.mean_rev)
mean_rev = config_postmean.mean_rev
ts_step = 1 / ts_length

# In[3]:


postMeanScoreModel = ConditionalLSTMTSPostMeanScoreMatching(*config_postmean.model_parameters)
postMeanScoreModel.load_state_dict(torch.load(config_postmean.scoreNet_trained_path + "_NEp" + str(train_epoch)))

rpostMeanScoreModel = ConditionalLSTMTSPostMeanScoreMatching(*config_rpostmean.model_parameters)
rpostMeanScoreModel.load_state_dict(torch.load(config_rpostmean.scoreNet_trained_path + "_NEp" + str(train_epoch)))

rrpostMeanScoreModel = ConditionalLSTMTSPostMeanScoreMatching(*config_rrpostmean.model_parameters)
rrpostMeanScoreModel.load_state_dict(torch.load(config_rrpostmean.scoreNet_trained_path + "_NEp" + str(train_epoch)))

rrrpostMeanScoreModel = ConditionalLSTMTSPostMeanScoreMatching(*config_rrrpostmean.model_parameters)
rrrpostMeanScoreModel.load_state_dict(torch.load(config_rrrpostmean.scoreNet_trained_path + "_NEp" + str(train_epoch)))

rrrrpostMeanScoreModel = ConditionalLSTMTSPostMeanScoreMatching(*config_rrrrpostmean.model_parameters)
rrrrpostMeanScoreModel.load_state_dict(
    torch.load(config_rrrrpostmean.scoreNet_trained_path + "_NEp" + str(train_epoch)))

scoreScoreModel = ConditionalLSTMTSScoreMatching(*config_score.model_parameters)
scoreScoreModel.load_state_dict(torch.load(config_score.scoreNet_trained_path + "_NEp" + str(train_epoch)))

scaledScoreScoreModel = ConditionalLSTMTSScoreMatching(*config_score_scaled.model_parameters)
scaledScoreScoreModel.load_state_dict(torch.load(config_score_scaled.scoreNet_trained_path + "_NEp" + str(train_epoch)))


# In[4]:


def single_time_sampling(config, diff_time_space, diffusion, feature, scoreModel, device, prev_path):
    x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
    scores = []
    exp_scores = []
    revSDE_paths = []
    for diff_index in tqdm(range(config.max_diff_steps)):
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
        exp_const = torch.sqrt(diffusion_mean2) * (ts_step) * (-config.mean_rev * prev_path.squeeze(-1))
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
        # Now update sample
        z = torch.randn_like(drift)
        x = drift + diffParam * z
    scores = torch.flip(torch.concat(scores, dim=-1), dims=[1])
    exp_scores = torch.flip(torch.concat(exp_scores, dim=-1), dims=[1])
    revSDE_paths = torch.flip(torch.concat(revSDE_paths, dim=-1), dims=[1])
    # assert(scores.shape == (data_shape[0], config.max_diff_steps) and exp_scores.shape == (data_shape[0], config.max_diff_steps) and revSDE_paths == (data_shape[0], config.max_diff_steps))
    return x, scores, exp_scores, revSDE_paths


# In[5]:


# I want to compare the score errors for the different models, so run recursive diffusion for 2 steps only, and output the different scores
def run_whole_ts_recursive_diffusion(config, ts_length, initial_feature_input, diffusion, scoreModel, device,
                                     diff_time_scale, real_time_scale):
    stored_scores = []
    stored_expscores = []
    stored_revSDE_paths = []
    prev_paths = []
    samples = initial_feature_input
    cumsamples = initial_feature_input
    for t in (range(ts_length)):
        prev_paths.append(cumsamples)
        print("Sampling at real time {}\n".format(t + 1))
        if t == 0:
            feature, (h, c) = scoreModel.rnn(samples, None)
        else:
            feature, (h, c) = scoreModel.rnn(samples, (h, c))
        new_samples, scores, exp_scores, revSDE_paths = single_time_sampling(config=config,
                                                                             diff_time_space=diff_time_scale,
                                                                             diffusion=diffusion, scoreModel=scoreModel,
                                                                             device=device, feature=feature,
                                                                             prev_path=samples)
        cumsamples = samples + new_samples
        samples = new_samples  # But we feed latest INCREMENT to the LSTM
        stored_scores.append(scores.unsqueeze(1))
        stored_expscores.append(exp_scores.unsqueeze(1))
        stored_revSDE_paths.append(revSDE_paths.unsqueeze(1))
    stored_scores = torch.concat(stored_scores, dim=1)
    # assert(stored_scores.shape == (data_shape[0], T, config.max_diff_steps))
    stored_expscores = torch.concat(stored_expscores, dim=1)
    # assert(stored_expscores.shape == (data_shape[0], T, config.max_diff_steps))
    stored_revSDE_paths = torch.concat(stored_revSDE_paths, dim=1)
    # assert(stored_revSDE_paths.shape == (data_shape[0], T, config.max_diff_steps))
    prev_paths = torch.concat(prev_paths, dim=1).squeeze(-1)
    print((stored_scores.shape, stored_expscores.shape, stored_revSDE_paths.shape, prev_paths.shape))
    return stored_scores, stored_expscores, stored_revSDE_paths, prev_paths


# In[7]:

# Build drift estimator
def build_drift_estimator(ts_step, ts_length, diff_time_space, score_evals, Xtaus, prev_paths):
    eff_times = diffusion.get_eff_times(torch.Tensor(diff_time_space)).numpy()
    beta_2_taus = np.exp(-eff_times)
    sigma_taus = 1. - beta_2_taus
    # Compute the part of the score independent of data mean
    c1 = (sigma_taus + beta_2_taus * ts_step) * np.exp(0.5 * eff_times)  # * 1/beta_tau
    c2 = np.exp(0.5 * eff_times)  # 1/beta_tau
    mean_est = c1 * score_evals + (c2.reshape(1, 1, -1)) * Xtaus
    mean_est /= ts_step
    print(mean_est.shape)
    true_drifts = (-1 * mean_rev * prev_paths)
    for t in range(ts_length):
        # Check drifts make sense
        plt.plot(prev_paths[:, t], true_drifts[:, t])
        plt.title("True Drift Against Previous State")
        plt.show()
        plt.close()
        curr_time_mean_ests = mean_est[:, t, :]
        curr_time_true_drift = true_drifts[:, t]
        print(curr_time_mean_ests.shape, true_drifts.shape)
        # Plot MSE of drift as a function of diffusion time
        mses = np.mean(np.power(curr_time_mean_ests - curr_time_true_drift[:, np.newaxis], 2), 0)
        plt.plot(diff_time_space, mses)
        plt.title(f"Drift MSE as a function of diffusion time at real time {t + 1}")
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:400], mses[50:400])
        plt.title(f"Drift MSE as a function of diffusion time first 50 to 400 at real time {t + 1}")
        plt.show()
        plt.close()
        plt.plot(diff_time_space[7900:], mses[7900:])
        plt.title(f"Drift MSE as a function of diffusion time last 1100 at real time {t + 1}")
        plt.show()
        plt.close()
        plt.plot(diff_time_space[4000:7900], mses[4000:7900])
        plt.title(f"Drift MSE as a function of diffusion time 4000:7900 at real time {t + 1}")
        plt.show()
        plt.close()
        print(np.min(mses), np.argmin(mses))
        argmin = np.argmin(mses)
    # Choose a diffusion time for the mean estimator
    best_mean_est = mean_est[:, :, argmin]
    for t in range(ts_length):
        paired = zip(prev_paths[:, t], best_mean_est[:, t])
        sorted_pairs = sorted(paired, key=lambda x: x[0])

        # Separate the pairs back into two arrays
        currprevpath, currmeanest = zip(*sorted_pairs)
        currprevpath = np.array(currprevpath)
        currmeanest = np.array(currmeanest)
        plt.scatter(currprevpath, currmeanest, label="Sorted Estimated Drift", s=1.5)
        plt.scatter(currprevpath, -mean_rev * currprevpath, label="True Drift", s=1.5)
        plt.title(f"Est/True Drift Against State at real time {t + 1}")
        plt.show()
        plt.close()

    return mean_est


def analyse_score_models(config, ts_length, max_diff_steps, sample_eps, diffusion, ts_step, mean_rev, scores,
                         exp_scores, prev_paths, revSDE_paths, modeltype):
    # First plot the score errors for one of the models
    diff_time_space = np.linspace(sample_eps, 1, max_diff_steps)
    # Finally, build drift estimator
    # build_drift_estimator(ts_step=ts_step, ts_length=ts_length, diff_time_space=diff_time_space, score_evals=scores,
    #                      Xtaus=revSDE_paths, prev_paths=prev_paths)

    score_errs = np.mean(np.power(scores - exp_scores, 2), axis=0)
    print(score_errs.shape)
    assert (score_errs.shape == (ts_length, max_diff_steps))
    for t in range(ts_length):
        plt.plot(diff_time_space, score_errs[t, :], label=modeltype)
        plt.title(f"MSE Scores at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:400], score_errs[t, 50:400], label=modeltype)
        plt.title(f"MSE First 50to400 Scores at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8000:], score_errs[t, 8000:], label=modeltype)
        plt.title(f"MSE Last 2000 Scores at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()

    # Which part of the score is the issue?
    eff_times = diffusion.get_eff_times(torch.Tensor(diff_time_space)).numpy()
    beta_taus = np.exp(-0.5 * eff_times)
    beta_2_taus = np.exp(-eff_times)
    sigma_taus = 1. - beta_2_taus

    # View the reverse diffusion drift score error
    rev_diff_drift = np.mean(np.power(scores * sigma_taus, 1), axis=0)
    assert (rev_diff_drift.shape == (ts_length, max_diff_steps))
    for t in range(ts_length):
        plt.plot(diff_time_space, rev_diff_drift[t, :], label=modeltype)
        plt.title(f"RevDiff Drift at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:400], rev_diff_drift[t, 50:400], label=modeltype)
        plt.title(f"RevDiff Drift 50to400 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8000:], rev_diff_drift[t, 8000:], label=modeltype)
        plt.title(f"RevDiff Drift Last 2000 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()

    rev_diff_drift = np.mean(np.power(exp_scores * sigma_taus, 1), axis=0)
    assert (rev_diff_drift.shape == (ts_length, max_diff_steps))
    for t in range(ts_length):
        plt.plot(diff_time_space, rev_diff_drift[t, :], label=modeltype)
        plt.title(f"Expected RevDiff Drift at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:400], rev_diff_drift[t, 50:400], label=modeltype)
        plt.title(f"Expected RevDiff Drift 50to400 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8000:], rev_diff_drift[t, 8000:], label=modeltype)
        plt.title(f"Expected RevDiff Drift Last 2000 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()

    # Compute the part of the score independent of data mean
    c1 = -np.power(sigma_taus + beta_2_taus * ts_step, -1)
    exp_Xtau_score_component = c1 * revSDE_paths
    # Compute the part of the score dependent only on the data mean
    c2 = -beta_taus * c1
    data_means = (ts_step * -1 * mean_rev * prev_paths)[:, :, np.newaxis]
    exp_dataMean_score_component = data_means * (c2.reshape(1, 1, -1))
    # Check the sum of both parts is the same as the expected scores
    for t in range(0):
        Xtau_component = exp_scores[:, t, :] - exp_dataMean_score_component[:, t, :]
        dataMean_component = exp_scores[:, t, :] - exp_Xtau_score_component[:, t, :]
        plt.plot(diff_time_space, np.mean(np.power(Xtau_component - exp_Xtau_score_component[:, t, :], 2), 0))
        plt.title(f"MSE dataMean Independent Component agrees real time {t + 1}")
        # plt.show()
        plt.close()
        plt.plot(diff_time_space, np.mean(np.power(dataMean_component - exp_dataMean_score_component[:, t, :], 2), 0))
        plt.title(f"MSE dataMean dependent Component agrees real time {t + 1}")
        # plt.show()
        plt.close()
    # Now check each component of the score individually
    for t in range(0):
        Xtau_component = scores[:, t, :] - exp_dataMean_score_component[:, t, :]
        plt.plot(diff_time_space[0:], np.mean(np.power(Xtau_component - exp_Xtau_score_component[:, t, :], 2), 0)[0:],
                 label="Estimated DataMean-Independent")
        plt.title(f"MSE Estimated DataMean Independent Component at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:], np.mean(np.power(Xtau_component - exp_Xtau_score_component[:, t, :], 2), 0)[50:],
                 label="Estimated DataMean-Independent")
        plt.title(f"MSE Estimated DataMean Independent Component After 50 at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8500:],
                 np.mean(np.power(Xtau_component - exp_Xtau_score_component[:, t, :], 2), 0)[8500:],
                 label="Estimated DataMean-Independent")
        plt.title(f"MSE Estimated DataMean Independent Component After 8500 at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
    for t in range(0):
        dataMean_component = scores[:, t, :] - exp_Xtau_score_component[:, t, :]
        plt.plot(diff_time_space[:],
                 np.mean(np.power(dataMean_component - exp_dataMean_score_component[:, t, :], 2), 0)[0:],
                 label="Estimated DataMean-Dependent")
        plt.title(f"MSE Estimated DataMean Dependent Component at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:],
                 np.mean(np.power(dataMean_component - exp_dataMean_score_component[:, t, :], 2), 0)[50:],
                 label="Estimated DataMean-Dependent")
        plt.title(f"MSE Estimated DataMean Dependent Component After 50 at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8500:],
                 np.mean(np.power(dataMean_component - exp_dataMean_score_component[:, t, :], 2), 0)[8500:],
                 label="Estimated DataMean-Dependent")
        plt.title(f"MSE Estimated DataMean Dependent Component After 8500 at time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()

    # Compute first the posterior mean
    data_mean_comp = (sigma_taus / (sigma_taus + beta_2_taus * ts_step)) * data_means
    Xtau_comp = (beta_taus * ts_step / (sigma_taus + beta_2_taus * ts_step)) * revSDE_paths
    post_mean = data_mean_comp + Xtau_comp
    # Visualise relative components across diffusion time
    for t in range(0):  # ts_length
        plt.plot(diff_time_space, np.mean(data_mean_comp[:, t, :], 0), label="DataMeanComp")
        plt.title(f"True DataMean Comp of Post Mean at real time {t + 1}")
        # plt.show()
        plt.close()
        plt.plot(diff_time_space, np.mean(Xtau_comp[:, t, :], 0), label="XtauComp")
        plt.title(f"True Xtau Comp of Post Mean at real time {t + 1}")
        # plt.show()
        plt.close()
    # Check posterior mean from the expected score agrees with the computed posterior mean
    for t in range(0):  # ts_length
        realtime_exp_scores = exp_scores[:, t, :]
        realtime_Xtaus = revSDE_paths[:, t, :]
        exp_post_mean = (-sigma_taus * realtime_exp_scores - realtime_Xtaus) / (-beta_taus)
        true_post_mean = post_mean[:, t, :]
        plt.plot(diff_time_space, np.mean(np.power(exp_post_mean - true_post_mean, 2), 0), label="PostMeanFromExpScore")
        plt.title(f"MSE of Posterior Mean from Expected Score at real time {t + 1}")
        plt.legend()
        # plt.show()
        plt.close()

    # Now check posterior mean according to score network
    for t in range(ts_length):
        realtime_scores = scores[:, t, :]
        realtime_Xtaus = revSDE_paths[:, t, :]
        realtime_network_evals = (-sigma_taus * realtime_scores - realtime_Xtaus) / (-beta_taus)
        realtime_post_mean = post_mean[:, t, :]
        plt.plot(diff_time_space[0:], np.mean(np.power(realtime_network_evals, 1), 0)[0:], label="Evaluations")
        plt.plot(diff_time_space[0:], np.mean(np.power(realtime_post_mean, 1), 0)[0:], label="True")
        plt.title(f"Estimated vs True Posterior Mean at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[0:], np.mean(np.power(realtime_network_evals - realtime_post_mean, 2), 0)[0:],
                 label="True")
        plt.title(f"MSE of Estimated Posterior Mean at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[50:], np.mean(np.power(realtime_network_evals - realtime_post_mean, 2), 0)[50:],
                 label="True")
        plt.title(f"MSE of Estimated Posterior Mean After 50 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[:50], np.mean(np.power(realtime_network_evals - realtime_post_mean, 2), 0)[:50],
                 label="True")
        plt.title(f"MSE of Estimated Posterior Mean First 50 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[8500:], np.mean(np.power(realtime_network_evals - realtime_post_mean, 2), 0)[8500:],
                 label="True")
        plt.title(f"MSE of Estimated Posterior Mean After 8500 at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        # Now check posterior mean scaled by the relevant factor
        realtime_scaled_network_evals = realtime_network_evals * (beta_taus / sigma_taus)
        realtime_scaled_post_mean = post_mean[:, t, :] * (beta_taus / sigma_taus)
        plt.plot(diff_time_space[0:], np.mean(np.power(realtime_scaled_network_evals, 1), 0)[0:], label="Evaluations")
        plt.plot(diff_time_space[0:], np.mean(np.power(realtime_scaled_post_mean, 1), 0)[0:], label="True")
        plt.title(f"Scaled Estimated vs True Posterior Mean at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()
        plt.plot(diff_time_space[0:],
                 np.mean(np.power(realtime_scaled_network_evals - realtime_scaled_post_mean, 2), 0)[0:],
                 label="Evaluations")
        plt.title(f"MSE of Scaled Estimated Posterior Mean at real time {t + 1}")
        plt.legend()
        plt.show()
        plt.close()


# In[10]:
T = 3

# Experiment for postmean score model
initial_feature_input = torch.zeros(data_shape).to(device)
postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_postmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=postMeanScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_postmean, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=postMean_scores.numpy(),
                     exp_scores=postMean_expscores.numpy(), revSDE_paths=postMean_revSDEpaths.numpy(),
                     prev_paths=postMean_prevPaths.numpy(), modeltype="postMean")
"""
# Experiment for rpostmean score model
initial_feature_input = torch.zeros(data_shape).to(device)
postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_rpostmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=rpostMeanScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_rpostmean, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=postMean_scores.numpy(),
                     exp_scores=postMean_expscores.numpy(), revSDE_paths=postMean_revSDEpaths.numpy(),
                     prev_paths=postMean_prevPaths.numpy(), modeltype="rpostMean")

# Experiment for rrpostmean score model
initial_feature_input = torch.zeros(data_shape).to(device)
postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_rrpostmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=rrpostMeanScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_rrpostmean, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=postMean_scores.numpy(),
                     exp_scores=postMean_expscores.numpy(), revSDE_paths=postMean_revSDEpaths.numpy(),
                     prev_paths=postMean_prevPaths.numpy(), modeltype="rrpostMean")


# Experiment for rrrpostmean score model
initial_feature_input = torch.zeros(data_shape).to(device)
postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_rrrpostmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=rrrpostMeanScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_rrrpostmean, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=postMean_scores.numpy(),
                     exp_scores=postMean_expscores.numpy(), revSDE_paths=postMean_revSDEpaths.numpy(),
                     prev_paths=postMean_prevPaths.numpy(), modeltype="rrrpostMean")


# Experiment for rrrrpostmean score model
initial_feature_input = torch.zeros(data_shape).to(device)
postMean_scores, postMean_expscores, postMean_revSDEpaths, postMean_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_rrrrpostmean, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=rrrrpostMeanScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_rrrrpostmean, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=postMean_scores.numpy(),
                     exp_scores=postMean_expscores.numpy(), revSDE_paths=postMean_revSDEpaths.numpy(),
                     prev_paths=postMean_prevPaths.numpy(), modeltype="rrrrpostMean")

# Experiment for scaled score model
initial_feature_input = torch.zeros(data_shape).to(device)
scaledScore_scores, scaledScore_expscores, scaledScore_revSDEpaths, scaledScore_prevPaths = run_whole_ts_recursive_diffusion(
    ts_length=T, config=config_score_scaled, initial_feature_input=initial_feature_input, diffusion=diffusion,
    scoreModel=scaledScoreScoreModel, device=device, diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

analyse_score_models(config=config_score_scaled, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=scaledScore_scores.numpy(),
                     exp_scores=scaledScore_expscores.numpy(), revSDE_paths=scaledScore_revSDEpaths.numpy(),
                     prev_paths=scaledScore_prevPaths.numpy(), modeltype="scaledScore")



# Experiment for score score model
initial_feature_input = torch.zeros(data_shape).to(device)
scores_scores, scores_expscores, scores_revSDEpaths, scores_prevPaths = run_whole_ts_recursive_diffusion(ts_length=T,
                                                                                                         config=config_score,
                                                                                                         initial_feature_input=initial_feature_input,
                                                                                                         diffusion=diffusion,
                                                                                                         scoreModel=scoreScoreModel,
                                                                                                         device=device,
                                                                                                         diff_time_scale=diff_time_scale,
                                                                                                         real_time_scale=real_time_scale)

analyse_score_models(config=config_score, ts_length=T, max_diff_steps=max_diff_steps, sample_eps=sample_eps,
                     ts_step=ts_step, mean_rev=mean_rev, diffusion=diffusion, scores=scores_scores.numpy(),
                     exp_scores=scores_expscores.numpy(), revSDE_paths=scores_revSDEpaths.numpy(),
                     prev_paths=scores_prevPaths.numpy(), modeltype="scores")

"""
