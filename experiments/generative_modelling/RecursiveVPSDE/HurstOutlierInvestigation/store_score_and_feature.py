import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import cleanup_experiment, init_experiment
from utils.math_functions import compute_fBm_cov, compute_fBn_cov


def recursive_sampling_and_track(data_shape: tuple, torch_device, feature: torch.Tensor,
                                 diffusion: VPSDEDiffusion,
                                 scoreModel: ConditionalTimeSeriesScoreMatching,
                                 config: ConfigDict, ctvar: torch.Tensor, cv1:torch.Tensor, cv2:torch.Tensor, true_past:torch.Tensor):
    """
    Run through whole ancestral sampling for single real time index
    :param data_shape: Size of data output
    :param torch_device: Pytorch device
    :param feature: Feature matrix
    :param diffusion: Diffusion model
    :param scoreModel: Trained score-matching network
    :param config: Experiment configuration file
    :param data_cov: True data covariance matrix
    :return:
        1. Sample random variables
        2. Drift Error throughout ancestral sampling
    """
    x = diffusion.prior_sampling(shape=(data_shape[0],1,data_shape[-1])).to(torch_device)  # Move to correct device
    x_true = diffusion.prior_sampling(shape=(data_shape[0],1, data_shape[-1])).to(torch_device)  # Move to correct device
    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps,
                               steps=config.max_diff_steps).to(torch_device)
    drift_errors = torch.zeros(size=(config.max_diff_steps, config.dataSize))
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False, desc="Sampling :: ", position=0):
        diff_index = torch.Tensor([i]).to(torch_device).long()
        t = timesteps[diff_index]
        # Obtain required diffusion parameters
        pred_score, pred_drift, diffusion_param = diffusion.get_conditional_ancestral_sampling(x, t=t * torch.ones(
            (x.shape[0], )).to(torch_device), feature=feature, score_network=scoreModel, diff_index=diff_index,
                                                                                               max_diff_steps=config.max_diff_steps)

        eff_time = diffusion.get_eff_times(diff_times=t)
        exp_var = (ctvar - cv1@cv2)*torch.ones_like(x)
        exp_mean = ((1. - torch.exp(-0.5*eff_time))*cv1@true_past)
        assert(exp_mean.shape == (config.dataSize, 1, 1))
        inv_cov = - torch.linalg.inv((1. - torch.exp(-eff_time)) + torch.exp(-eff_time) * exp_var)
        beta_t = diffusion.get_discretised_beta(config.max_diff_steps - 1 - diff_index, config.max_diff_steps)
        exp_drift = x * (2. - torch.sqrt(1. - beta_t)) + (beta_t * (torch.matmul(inv_cov, (x-exp_mean))))

        # One-step reverse-time SDE
        x = pred_drift + diffusion_param * torch.randn_like(x)
        x_true = exp_drift + diffusion_param*torch.randn_like(x)
        drift_errors[config.max_diff_steps - 1 - i, :] = torch.pow(
            torch.linalg.norm((pred_drift - exp_drift).squeeze(1).T, ord=2, axis=0),
            2).detach().cpu()
    return x, x_true, drift_errors


@record
def run_feature_drift_recursive_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalTimeSeriesScoreMatching, data_shape,
                                         config: ConfigDict, rng: np.random.Generator):
    """
    Recursive reverse sampling using LSTMs and tracking feature and drift values
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    assert (config.predictor_model == "ancestral")

    if config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)

    features = []
    scoreModel.eval()
    scoreModel.to(device)
    with torch.no_grad():
        samples = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
        paths = []
        true_paths = torch.zeros(size=(data_shape[0], config.timeDim, data_shape[-1])).to(device)
        drift_errors = []
        for t in range(config.timeDim):
            print("Sampling at real time {}\n".format(t + 1))
            if t == 0:
                output, (h, c) = scoreModel.rnn(samples, None)
                true_past = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
                curr_time_cov1 = torch.zeros(size=(1, 1)).to(device)
                curr_time_cov2 = torch.zeros(size=(1, 1)).to(device)
                curr_var = torch.Tensor([1./(config.timeDim**(2*config.hurst))]).to(device)
                assert(data_cov[0,0] == curr_var)
            else:
                output, (h, c) = scoreModel.rnn(samples, (h, c))
                true_past = true_paths[:,:t,:]#torch.concat(true_paths, dim=1).to(device)
                curr_time_cov1 = torch.atleast_2d(data_cov[t,:t]@torch.linalg.inv(data_cov[:t,:t])).to(device)
                curr_time_cov2 = torch.atleast_2d((data_cov[:t,t])).T.to(device)
                curr_var = torch.atleast_2d(data_cov[t, t]).to(device)
                assert(true_past.shape == (config.dataSize, t, 1) and curr_time_cov1.shape==(1, t) and curr_time_cov2.shape==(t,1))
            samples, true_samples, per_time_drift_error = recursive_sampling_and_track(data_shape=data_shape, torch_device=device, feature=output,
                                 diffusion=diffusion,
                                 scoreModel=scoreModel,
                                 config=config, ctvar=curr_var, cv1=curr_time_cov1, cv2=curr_time_cov2, true_past=true_past)
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
            true_paths[:,[t],:] = true_samples
            features.append(output.permute(1,0,2))
            drift_errors.append(per_time_drift_error.unsqueeze(0))

    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)
    feature_df = torch.concat(features, dim=0).cpu()
    assert (feature_df.shape == (config.timeDim, config.dataSize, 40))
    drift_error_df = torch.concat(drift_errors, dim=0).cpu()
    assert (drift_error_df.shape == (config.timeDim, config.max_diff_steps, config.dataSize))
    return np.atleast_2d(final_paths.numpy()), np.atleast_3d(feature_df.numpy()), np.atleast_3d(drift_error_df.numpy())


def store_score_and_feature() -> None:
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)

    scoreModel = ConditionalTimeSeriesScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)
    config.dataSize = 5000000
    config.max_diff_steps = 5

    train_epoch = 1920
    assert (train_epoch in config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(train_epoch)))
    except FileNotFoundError as e:
        assert FileNotFoundError(
            "Error {}; no valid trained model found; train before initiating experiment\n".format(e))
    cleanup_experiment()
    rng = np.random.default_rng()
    paths, features, drift_errors = run_feature_drift_recursive_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                                                         data_shape=(
                                                                         config.dataSize, config.timeDim, 1),
                                                                         config=config, rng=rng)
    assert(paths.shape == (config.dataSize, config.timeDim) and features.shape == (config.timeDim, config.dataSize, 40),
    drift_errors.shape == (config.timeDim, config.max_diff_steps, config.dataSize))
    print("Storing Drift Errors\n")
    # Store
    drift_data_path = config.experiment_path.replace("results/",
                                                     "results/drift_data/") + "_Nepochs{}_SFS".format(train_epoch).replace(
        ".", "") + ".csv.gzip"
    drift_df = pd.concat({i:pd.DataFrame(drift_errors[i, :, :]) for i in tqdm(range(config.timeDim))})
    print(drift_df)
    drift_df.index = pd.MultiIndex.from_product([np.arange(0, config.timeDim), np.arange(0, config.max_diff_steps)]).set_names(["Time", "DiffTime"], inplace=False)
    drift_df.info()
    drift_df.to_csv(drift_data_path, compression="gzip")
    del drift_df
    print("Done Storing Drift Errors\n")

    print("Storing Path Data\n")
    path_df = pd.DataFrame(paths)
    path_df.index = pd.MultiIndex.from_product(
        [["Final Time Samples"], [i for i in range(config.dataSize)]])
    path_df.to_csv(config.experiment_path + "_Nepochs{}_SFS.csv.gzip".format(train_epoch), compression="gzip")
    path_df.info()
    del path_df
    print("Done Storing Path Data\n")

    print("Storing Feature Data\n")
    feature_data_path = config.experiment_path.replace("results/", "results/feature_data/") + "_Nepochs{}_SFS".format(
        train_epoch).replace(".", "") + ".csv.gzip"
    feature_df = pd.concat([pd.DataFrame(features[i, :, :]) for i in range(config.timeDim)])
    feature_df.index = pd.MultiIndex.from_product([np.arange(0, config.timeDim), np.arange(0, config.dataSize)]).set_names(["Time", "Sample Id"], inplace=False)
    feature_df.info()
    feature_df.to_csv(feature_data_path, compression="gzip")
    del feature_df
    print("Storing Feature Data\n")

if __name__ == "__main__":
    store_score_and_feature()
