#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from configs.RecursiveVPSDE.recursive_PostMeanScore_fOU_T256_H07_tl_5data import get_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching

# In[3]:


config = get_config()
assert (0 < config.hurst < 1.)
assert (config.early_stop_idx == 0)
assert (config.tdata_mult == 5)
print(config.scoreNet_trained_path, config.dataSize)
rng = np.random.default_rng()
scoreModel = ConditionalLSTMTSPostMeanScoreMatching(
    *config.model_parameters) if config.model_choice == "TSM" else None
train_epoch = 960
scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
N = 500
data_shape = (N, 1, 1)
device = "cpu"
diff_time_scale = torch.linspace(start=config.end_diff_time, end=config.sample_eps,
                                 steps=config.max_diff_steps)
diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
ts_step = 1 / config.ts_length

# In[4]:


# real time 0
prev_path = torch.zeros((data_shape[0], 1, data_shape[-1])).to(
    device)  # torch.normal(mean=0, std=np.sqrt(1/config.ts_length), size=(data_shape[0], 1, data_shape[-1])).to(device)
feature, (h, c) = scoreModel.rnn(prev_path, None)  # Feature for real time 0
tau = diff_time_scale[900] * torch.ones((data_shape[0],)).to(device)  # End of reverse-diffusion

# In[5]:


# Create a linear sequence of "corrupted samples"
diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=tau))).T.to(device)
diffusion_var = 1. - diffusion_mean2
exp_mean = torch.sqrt(diffusion_mean2) * (ts_step) * (-0.8 * prev_path.squeeze(-1))
assert (exp_mean.shape == prev_path.squeeze(-1).shape)
exp_var = diffusion_var + diffusion_mean2 * (ts_step)
Xtaus = torch.sort(torch.normal(mean=exp_mean, std=torch.sqrt(exp_var)).reshape((N, data_shape[1], 1)), dim=0)[0]

# In[6]:


plt.hist(Xtaus.squeeze())
print(np.std(Xtaus.squeeze().numpy()), torch.sqrt(exp_var))

# In[7]:


with torch.no_grad():
    try:
        score_evals = scoreModel.forward(inputs=Xtaus, times=tau, conditioner=feature,
                                         eff_times=diffusion.get_eff_times(tau.reshape(Xtaus.shape)))
    except TypeError:
        score_evals = scoreModel.forward(inputs=Xtaus, times=tau, conditioner=feature)
score_evals.shape


# In[8]:


def check_linear_score(config, Xtaus, score_evals, diffusion_var, diffusion_mean2, ts_step, prev_path):
    assert (Xtaus.shape == (data_shape[0], 1, 1), score_evals.shape == (data_shape[0], 1, 1))
    Xts = Xtaus.squeeze()
    scores = score_evals.squeeze()
    paired = zip(Xts.numpy(), scores.numpy())
    # Sort the pairs based on values of arr1
    sorted_pairs = sorted(paired, key=lambda x: x[0])
    # Separate the pairs back into two arrays
    Xts, scores = zip(*sorted_pairs)
    Xts = np.array(Xts)
    scores = np.array(scores)
    plt.scatter(Xts, scores, label="Empirical Score Against State", s=1.5)
    exp_slope = -(1 / ((diffusion_var + diffusion_mean2 * ts_step))[0]).numpy()[0]
    exp_const = (torch.sqrt(diffusion_mean2) * (ts_step) * (-config.mean_rev * prev_path.squeeze(-1))).numpy()[0]
    plt.scatter(Xts, exp_slope * (Xts - exp_const), color="orange", label="True Score Against State", s=1.5)
    plt.title("Score")
    plt.legend()
    plt.show()
    plt.close()
    plt.scatter(Xts, scores / exp_slope, label="Empirical", s=1.5)
    plt.scatter(Xts, Xts - exp_const, label="True", s=1.5)
    plt.title("Linear Term ONLY")
    plt.legend()
    plt.show()
    plt.close()
    plt.scatter(Xts, -Xts + scores / exp_slope, label="Empirical", s=1.5)
    plt.scatter(Xts, -exp_const * np.ones_like(Xts), label="True", s=1.5)
    plt.title("Offset Term ONLY")
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(Xts, label="Corrupted Samples")
    plt.title("Corrupted Samples in Score Against Sample Number")
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(exp_const * np.ones_like(Xts), label="Offset Term")
    plt.title("Offset Term in Slope Against Sample Number")
    plt.legend()
    plt.show()
    plt.close()


check_linear_score(config=config, Xtaus=Xtaus, score_evals=score_evals, diffusion_var=diffusion_var,
                   diffusion_mean2=diffusion_mean2, ts_step=ts_step, prev_path=prev_path)


# In[9]:


# Build drift estimator
def build_drift_estimator(config, diffusion_var, diffusion_mean2, ts_step, score_evals, Xtaus, prev_path):
    try:
        assert (len(diffusion_var.shape) == len(diffusion_mean2.shape) == len(score_evals.shape) == len(
            Xtaus.shape) == len(prev_path.shape))
    except AssertionError:
        diffusion_var = diffusion_var.squeeze().numpy()
        diffusion_mean2 = diffusion_mean2.squeeze().numpy()
        score_evals = score_evals.squeeze().numpy()
        Xtaus = Xtaus.squeeze().numpy()
        prev_path = prev_path.squeeze().numpy()
        assert (type(ts_step) == float)
    c1 = ((diffusion_var + diffusion_mean2 * ts_step) / (np.power(diffusion_mean2, 0.5)))
    c2 = (np.power(diffusion_mean2, -0.5))
    print(c1.shape, c2.shape, score_evals.shape, Xtaus.shape)
    mean_est = c1 * score_evals + c2 * Xtaus
    mean_est /= ts_step
    assert (prev_path.shape == mean_est.shape)

    paired = zip(prev_path, mean_est)
    # Sort the pairs based on values of arr1
    sorted_pairs = sorted(paired, key=lambda x: x[0])
    # Separate the pairs back into two arrays
    prev_path, mean_est = map(lambda x: np.array(x), zip(*sorted_pairs))
    plt.plot(prev_path, mean_est, label="Drift Estimates As a Function of X")
    plt.plot(prev_path, -config.mean_rev * prev_path, label="Expected Drift As a Function of X")
    plt.title(f"Drift Estimator Against State with MSE {np.mean(np.power(mean_est - -config.mean_rev * prev_path, 2))}")
    plt.legend()
    plt.plot()
    plt.show()
    plt.close()
    plt.plot(mean_est, label="Drift Estimates")
    plt.plot(-config.mean_rev * prev_path, label="Expected Drift")
    plt.title("Drift Estimator")
    plt.legend()
    plt.plot()
    plt.show()
    plt.close()
    return mean_est


mean_est = build_drift_estimator(config=config, diffusion_var=diffusion_var, diffusion_mean2=diffusion_mean2,
                                 ts_step=ts_step, score_evals=score_evals, Xtaus=Xtaus, prev_path=prev_path)


# In[10]:


# Plot some marginal distributions for estimated drift
def drift_fOU_marginals(config, means, paths):
    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
    for i in range(3):
        idx = np.random.randint(low=0, high=config.ts_length)
        mean = means[:, idx]  # -gamma*X(t-1)
        plt.hist(mean, bins=150, density=True, label="Estimated Drift")
        plt.legend()
        plt.show()
        plt.close()
        t = time_space[idx - 1]
        expmeanrev = np.exp(-config.mean_rev * t)
        exp_mean = 0 * (1. - expmeanrev)
        exp_mean += paths[:, idx - 1] * expmeanrev  # Initial state is the previous path
        exp_mean *= -config.mean_rev
        exp_var = np.power(1, 2)
        exp_var /= (2 * config.mean_rev)
        exp_var *= 1. - np.power(expmeanrev, 2)
        exp_var *= config.mean_rev * config.mean_rev
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=mean.shape[0])
        plt.hist(exp_rvs, bins=150, density=True, label="Expected Drift")
        plt.title(f"Marginal Distributions of Drift at time {t} for epoch {0}")
        plt.legend()
        plt.show()
        plt.close()


# In[11]:


def single_time_sampling(config, diff_time_space, diffusion, feature, scoreModel, device, prev_path, param_est_time):
    x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
    scores = []
    exp_scores = []
    for diff_index in tqdm(range(config.max_diff_steps)):
        tau = diff_time_space[diff_index] * torch.ones((data_shape[0],)).to(device)
        try:
            scoreModel.eval()
            with torch.no_grad():
                tau = tau * torch.ones((x.shape[0],)).to(device)
                predicted_score = scoreModel.forward(x, conditioner=feature, times=tau)
            score, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=x,
                                                                                  predicted_score=predicted_score,
                                                                                  diff_index=torch.Tensor(
                                                                                      [int(diff_index)]).to(device),
                                                                                  max_diff_steps=config.max_diff_steps)
        except TypeError as e:
            scoreModel.eval()
            with torch.no_grad():
                tau = tau * torch.ones((x.shape[0],)).to(device)
                eff_times = diffusion.get_eff_times(diff_times=tau)
                eff_times = eff_times.reshape(x.shape)
                predicted_score = scoreModel.forward(x, conditioner=feature, times=tau, eff_times=eff_times)
            score, drift, diffParam = diffusion.get_conditional_learnsample_reverse_diffusion(x=x,
                                                                                              predicted_score=predicted_score,
                                                                                              diff_index=torch.Tensor(
                                                                                                  [int(diff_index)]).to(
                                                                                                  device),
                                                                                              max_diff_steps=config.max_diff_steps)
        if len(score.shape) == 3 and score.shape[-1] == 1:
            score = score.squeeze(-1)
        scores.append(score)
        diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=tau))).T.to(device)
        diffusion_var = 1. - diffusion_mean2
        exp_slope = -(1 / ((diffusion_var + diffusion_mean2 * ts_step))[0])
        exp_const = torch.sqrt(diffusion_mean2) * (ts_step) * (-config.mean_rev * prev_path.squeeze(-1))
        exp_score = exp_slope * (x.squeeze(-1) - exp_const)
        if len(exp_score) == 3 and exp_score.shape[0] == 1:
            exp_score = exp_score.squeeze(-1)
        exp_scores.append(exp_score)
        if diff_index == param_est_time:
            print(
                f"Param Estimation at Diff Time {param_est_time} with BetaTau2 {(diffusion_mean2[0].squeeze())} and SigmaTau {(diffusion_var[0].squeeze())}")
            check_linear_score(config=config, Xtaus=x, score_evals=score, diffusion_var=diffusion_var,
                               diffusion_mean2=diffusion_mean2, ts_step=ts_step, prev_path=prev_path)
            mean_est = build_drift_estimator(config=config, diffusion_var=diffusion_var,
                                             diffusion_mean2=diffusion_mean2, ts_step=ts_step, score_evals=score,
                                             Xtaus=x, prev_path=prev_path)
        z = torch.randn_like(drift)
        x = drift + diffParam * z
    return x, mean_est, torch.flip(torch.concat(scores, dim=-1), dims=[1]), torch.flip(torch.concat(exp_scores, dim=-1),
                                                                                       dims=[1])


# In[12]:


def evaluate_ts_marginal_histogram(samples, ts_time, config, train_epoch):
    expmeanrev = np.exp(-config.mean_rev * ts_time)
    exp_mean = config.mean * (1. - expmeanrev)
    exp_mean += config.initState * expmeanrev
    exp_var = np.power(config.diffusion, 2)
    exp_var /= (2 * config.mean_rev)
    exp_var *= 1. - np.power(expmeanrev, 2)
    exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=samples.shape[0])
    plt.hist(samples.squeeze(), bins=150, density=True, label="True")
    plt.hist(exp_rvs, bins=150, density=True, label="Expected")
    plt.title(f"Marginal Distributions at time {ts_time} for epoch {train_epoch}")
    plt.legend()
    plt.show()
    plt.close()


def evaluate_scores(scores, exp_scores, diff_time_scale):
    assert (scores.shape == exp_scores.shape)
    plt.plot(diff_time_scale, scores.mean(dim=0).squeeze())
    plt.title("Sample-Averaged Empirical scores")
    plt.show()
    plt.close()
    plt.plot(diff_time_scale, exp_scores.mean(dim=0).squeeze())
    plt.title("Sample-Averaged True scores")
    plt.show()
    plt.close()
    errs = torch.mean(torch.pow((scores - exp_scores).squeeze(), 2), dim=0)
    plt.plot(diff_time_scale, errs.squeeze())
    plt.title("MSE Of Scores")
    plt.show()
    plt.close()


def evaluate_per_time_samples(samples, scores, exp_scores, ts_time, train_epoch, diff_time_scale, config):
    try:
        assert (samples.shape == (data_shape[0], 1))
    except AssertionError:
        samples = samples.squeeze(-1)
    evaluate_ts_marginal_histogram(samples=samples, ts_time=ts_time, train_epoch=train_epoch, config=config)
    evaluate_scores(scores=scores, exp_scores=exp_scores, diff_time_scale=diff_time_scale.numpy()[::-1])


# In[13]:


def run_whole_ts_recursive_diffusion(config, initial_feature_input, diffusion, scoreModel, device, diff_time_scale,
                                     real_time_scale):
    samples = initial_feature_input
    param_est_time = 10
    for t in (range(3)):
        print("Sampling at real time {}\n".format(t + 1))
        if t == 0:
            feature, (h, c) = scoreModel.rnn(samples, None)
        else:
            feature, (h, c) = scoreModel.rnn(samples, (h, c))
        new_samples, mean, scores, exp_scores = single_time_sampling(config=config, diff_time_space=diff_time_scale,
                                                                     diffusion=diffusion, scoreModel=scoreModel,
                                                                     device=device, feature=feature, prev_path=samples,
                                                                     param_est_time=param_est_time)
        cumsumsamples = new_samples + samples  # For evaluation, we need PATH VALUE, not increment
        samples = new_samples  # But we feed latest INCREMENT to the LSTM
        evaluate_per_time_samples(samples=cumsumsamples, scores=scores, exp_scores=exp_scores,
                                  ts_time=real_time_scale[t], train_epoch=train_epoch, diff_time_scale=diff_time_scale,
                                  config=config)
    return scores, exp_scores


# In[ ]:


initial_feature_input = torch.zeros(data_shape).to(device)
diff_time_scale = torch.linspace(start=config.end_diff_time, end=config.sample_eps,
                                 steps=config.max_diff_steps)
real_time_scale = torch.linspace(start=1 / config.ts_length, end=1,
                                 steps=config.ts_length)
scores, exp_scores = run_whole_ts_recursive_diffusion(config=config, initial_feature_input=initial_feature_input,
                                                      diffusion=diffusion, scoreModel=scoreModel, device=device,
                                                      diff_time_scale=diff_time_scale, real_time_scale=real_time_scale)

# In[ ]:


errs = torch.pow(scores - exp_scores, 2).mean(0)

# In[ ]:


plt.plot(diff_time_scale.numpy()[::-1], errs)

# In[ ]:


evaluate_scores(scores=scores, exp_scores=exp_scores, diff_time_scale=diff_time_scale.numpy()[::-1])

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
