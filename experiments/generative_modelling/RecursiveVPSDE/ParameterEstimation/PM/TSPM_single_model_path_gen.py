import os

import torch
from tqdm import tqdm

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.data_processing import init_experiment


def true_cond_mean(config, prev_path):
    if "fOU" in config.data_path:
        return (-config.mean_rev * (prev_path.squeeze(-1) - config.mean))
    else:
        return (config.mean_rev * torch.sin(prev_path.squeeze(-1)))


# Generate value of path at time "t" by running reverse diffusion
def single_time_sampling(config, data_shape, diff_time_space, diffusion, feature, scoreModel, device, es
                         ):
    x = diffusion.prior_sampling(shape=data_shape).to(device)  # Move to correct device
    for diff_index in tqdm(range(config.max_diff_steps)):
        if diff_index <= config.max_diff_steps - es - 1:
            tau = diff_time_space[diff_index] * torch.ones((data_shape[0],)).to(device)
            try:
                scoreModel.eval()
                with torch.no_grad():
                    tau = tau * torch.ones((x.shape[0],)).to(device)
                    predicted_score = scoreModel.forward(x, conditioner=feature, times=tau)
            except TypeError as e:
                print(e)
                scoreModel.eval()
                with torch.no_grad():
                    tau = tau * torch.ones((x.shape[0],)).to(device)
                    eff_times = diffusion.get_eff_times(diff_times=tau)
                    eff_times = eff_times.reshape(x.shape)
                    predicted_score = scoreModel.forward(x, conditioner=feature, times=tau, eff_times=eff_times)

            _, drift, diffParam = diffusion.get_conditional_reverse_diffusion(x=x,
                                                                              predicted_score=predicted_score,
                                                                              diff_index=torch.Tensor(
                                                                                  [int(diff_index)]).to(device),
                                                                              max_diff_steps=config.max_diff_steps)
            z = torch.randn_like(drift)
            x = drift + diffParam * z
        else:
            return x
    return x


# In[5]:

# Generate sample paths from [0, ts_length]
def run_whole_ts_recursive_diffusion(config, ts_length, initial_feature_input, diffusion, scoreModel, device,
                                     diff_time_scale, data_shape, es):
    paths = []
    cumsamples = initial_feature_input
    for t in (range(ts_length)):
        paths.append(cumsamples.cpu())
        print("Sampling at real time {}\n".format(t + 1))
        scoreModel.eval()
        with torch.no_grad():
            if t == 0:
                feature, (h, c) = scoreModel.rnn(initial_feature_input, None)
            else:
                feature, (h, c) = scoreModel.rnn(cumsamples, (h, c))
        new_samples = single_time_sampling(config=config, data_shape=data_shape,
                                           diff_time_space=diff_time_scale,
                                           diffusion=diffusion, scoreModel=scoreModel,
                                           device=device, feature=feature,
                                           es=es)
        cumsamples = cumsamples + new_samples
    paths.append(cumsamples.cpu())
    paths = torch.concat(paths, dim=1).squeeze(-1)
    return paths.cpu()


def TSPM_drift_eval():
    from configs.RecursiveVPSDE.LSTM_fSin.recursive_PostMeanScore_fSin_T256_H05_tl_5data import get_config
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

    Nepoch = 960
    assert (config.max_diff_steps == 10000 and config.beta_min == 0.)
    for es in [0, 5, 10, 20, 50, 100, 150, 200]:
        if "fOU" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_ES{es}_PathGen_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean}Mean_{config.max_diff_steps}DiffSteps").replace(
                ".", "")
        elif "fSin" in config.data_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_ES{es}_fSin_PathGen_{Nepoch}Nep_{config.loss_factor}LFactor_{config.mean_rev}MeanRev_{config.max_diff_steps}DiffSteps").replace(
                ".", "")

        # Fix the number of training epochs and training loss objective loss
        PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters).to(device)
        PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))

        print(Nepoch, config.data_path, es, config.scoreNet_trained_path)
        # Fix the number of real times to run diffusion
        eval_ts_length = int(1.3 * config.ts_length)
        # Experiment for score model with fixed (Nepochs, loss scaling, drift eval time, Npaths simulated)
        initial_feature_input = torch.zeros(data_shape).to(device)
        paths = run_whole_ts_recursive_diffusion(
            ts_length=eval_ts_length, config=config, initial_feature_input=initial_feature_input, diffusion=diffusion,
            scoreModel=PM, device=device, diff_time_scale=revDiff_time_scale, data_shape=data_shape, es=es)

        torch.save(paths, save_path + "_paths")


if __name__ == "__main__":
    TSPM_drift_eval()
