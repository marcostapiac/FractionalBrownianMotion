import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from tqdm import tqdm

from experiments.generative_modelling.estimate_fSDEs import second_order_estimator, estimate_hurst_from_filter
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSScoreMatching import \
    ConditionalLSTMTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTSScoreMatching import \
    ConditionalTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP


def recursive_sampling_and_track(data_shape: tuple, torch_device, feature: torch.Tensor,
                                 diffusion: VPSDEDiffusion,
                                 scoreModel: ConditionalTSScoreMatching,
                                 config: ConfigDict, sampling: str, prev_state: torch.Tensor):
    """
    Run through whole ancestral sampling for single real time index
    :param data_shape: Size of data output
    :param torch_device: Pytorch device
    :param feature: Feature matrix
    :param diffusion: Diffusion model
    :param scoreModel: Trained score-matching network
    :param config: Experiment configuration file
    :param sampling: Sampling algorithm
    :param prev_state: Previous sample (used for expected score calculation)
    :return:
        1. Sample random variables
        2. Drift Error throughout ancestral sampling
    """
    x = diffusion.prior_sampling(shape=(data_shape[0], 1, data_shape[-1])).to(torch_device)  # Move to correct device
    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps,
                               steps=config.max_diff_steps).to(torch_device)
    score_errors = torch.zeros(size=(config.max_diff_steps, config.dataSize))
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False, desc="Sampling :: ", position=0):
        diff_index = torch.Tensor([i]).to(torch_device).long()
        t = timesteps[diff_index]
        # Obtain required diffusion parameters
        if sampling == "ancestral":
            pred_score, pred_drift, diffusion_param = diffusion.get_conditional_ancestral_sampling(x, t=t * torch.ones(
                (x.shape[0],)).to(torch_device), feature=feature, score_network=scoreModel, diff_index=diff_index,
                                                                                                   max_diff_steps=config.max_diff_steps)
        elif sampling == "reverse":
            pred_score, pred_drift, diffusion_param = diffusion.get_conditional_reverse_diffusion(x, t=t * torch.ones(
                (x.shape[0],)).to(torch_device), feature=feature, score_network=scoreModel, diff_index=diff_index,
                                                                                                  max_diff_steps=config.max_diff_steps)
        else:
            pred_score, pred_drift, diffusion_param = diffusion.get_conditional_probODE(x, t=t * torch.ones(
                (x.shape[0],)).to(torch_device), feature=feature, score_network=scoreModel, diff_index=diff_index,
                                                                                        max_diff_steps=config.max_diff_steps)

        # One-step reverse-time SDE
        diffusion_mean2 = torch.atleast_2d(torch.exp(-diffusion.get_eff_times(diff_times=t))).T.to(torch_device)
        diffusion_var = 1. - diffusion_mean2
        exp_score = -torch.pow(
            torch.Tensor([diffusion_var + diffusion_mean2 * (1. / config.ts_length)]).to(torch_device), -1) * (
                                x - torch.sqrt(diffusion_mean2) * (
                                    -config.mean_rev * prev_state * (1. / config.ts_length)))
        x = pred_drift + diffusion_param * torch.randn_like(x)
        score_errors[config.max_diff_steps - 1 - i, :] = torch.pow(
            torch.linalg.norm((pred_score - exp_score).squeeze(1).T, ord=2, axis=0),
            2).detach().cpu()
    return x, score_errors


@record
def run_feature_drift_recursive_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalTSScoreMatching, data_shape,
                                         config: ConfigDict, sampling: str):
    """
    Recursive reverse sampling using LSTMs and tracking feature and drift values
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :param sampling: Sampling algorithm
        :return: Final reverse-time samples
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    assert (config.predictor_model == "ancestral")

    features = []
    score_errors = []
    scoreModel.eval()
    scoreModel.to(device)
    with torch.no_grad():
        samples = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
        paths = [torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)]
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            if t == 0:
                output, (h, c) = scoreModel.rnn(samples, None)
            else:
                output, (h, c) = scoreModel.rnn(samples, (h, c))
            samples, per_time_score_error = recursive_sampling_and_track(data_shape=data_shape, torch_device=device,
                                                                         feature=output,
                                                                         diffusion=diffusion,
                                                                         scoreModel=scoreModel,
                                                                         config=config, sampling=sampling,
                                                                         prev_state=paths[-1])
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
            features.append(output.permute(1, 0, 2))
            score_errors.append(per_time_score_error.unsqueeze(0))

    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)[:, 1:].cumsum(axis=1)
    assert (final_paths.shape == (config.dataSize, config.ts_length))
    feature = torch.concat(features, dim=0).cpu()
    assert (feature.shape == (config.ts_length, config.dataSize, config.lstm_hiddendim))
    score_errors = torch.concat(score_errors, dim=0).cpu()
    assert (score_errors.shape == (config.ts_length, config.max_diff_steps, config.dataSize))
    return np.atleast_2d(final_paths.numpy()), np.atleast_3d(feature.numpy()), np.atleast_3d(score_errors.numpy())


def store_score_and_feature() -> None:
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config
    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    scoreModel = ConditionalLSTMTSScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    sampling = "reverse"
    train_epoch = 2920
    assert (train_epoch in config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
    except FileNotFoundError as e:
        assert FileNotFoundError(
            "Error {}; no valid trained model found; train before initiating experiment\n".format(e))
    paths, features, score_errors = run_feature_drift_recursive_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                                                         data_shape=(
                                                                             config.dataSize, config.ts_length, 1),
                                                                         config=config, sampling=sampling)
    assert (
            paths.shape == (config.dataSize, config.ts_length) and features.shape == (
        config.ts_length, config.dataSize, config.lstm_hiddendim) and score_errors.shape == (
            config.ts_length, config.max_diff_steps, config.dataSize))

    print("Storing Path Data\n")
    path_df = pd.DataFrame(paths)
    print(path_df)
    path_df_path = config.experiment_path + "_NEp{}.parquet.gzip".format(train_epoch)
    path_df.to_parquet(path_df_path, compression="gzip")
    path_df.info()
    U_a1, U_a2 = second_order_estimator(paths=path_df.to_numpy(), Nsamples=path_df.shape[0])
    hs = estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=train_epoch)
    hs = pd.DataFrame(hs, index=path_df.index)
    # Under-estimation
    lower = 0.45
    upper = 0.9
    bad_idxs_1 = hs.index[hs.lt(lower).any(axis=1)].to_list()  # Path IDs which satisfy condition
    bad_idxs_2 = hs.index[hs.gt(upper).any(axis=1)].to_list()  # Path IDs which satisfy condition
    good_idxs_1 = hs.index[hs.gt(lower).any(axis=1)].to_list()
    good_idxs_2 = hs.index[hs.lt(upper).any(axis=1)].to_list()
    good_idxs = (list(set(good_idxs_1) & set(good_idxs_2)))[:1000]
    del path_df
    print("Done Storing Path Data\n")

    print("Done Storing Drift Errors\n")

    print("Storing Feature Data\n")
    feature_data_path = (config.experiment_path.replace("results/", "results/feature_data/") + "_NEp{}".format(
        train_epoch).replace(".", "") + ".parquet.gzip").replace("rec_TSM_False_incs_True_unitIntv_", "")
    feature_df = pd.concat({i: pd.DataFrame(features[i, :, :]) for i in tqdm(range(config.ts_length))})
    print(feature_df)
    feature_df.info()
    bad_feat_df_1 = feature_df.loc[pd.IndexSlice[:, bad_idxs_1], :]
    bad_feat_df_1.to_parquet(feature_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"), compression="gzip")
    print(bad_feat_df_1)
    bad_feat_df_2 = feature_df.loc[pd.IndexSlice[:, bad_idxs_2], :]
    bad_feat_df_2.to_parquet(feature_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"), compression="gzip")
    print(bad_feat_df_2)
    good_feat_df = feature_df.loc[pd.IndexSlice[:, good_idxs], :]
    good_feat_df.to_parquet(feature_data_path.replace(".parquet.gzip", "_good.parquet.gzip"), compression="gzip")
    print(good_feat_df)
    # feature_df.to_parquet(feature_data_path, compression="gzip")
    del feature_df
    print("Done Storing Feature Data\n")

    print("Storing Score Errors\n")
    score_data_path = (config.experiment_path.replace("results/",
                                                      "results/score_errors/") + "_NEp{}".format(
        train_epoch).replace(
        ".", "") + ".parquet.gzip").replace("rec_TSM_False_incs_True_unitIntv_", "")
    score_df = pd.concat({i: pd.DataFrame(score_errors[i, :, :]) for i in tqdm(range(config.ts_length))})
    bad_score_df_1 = score_df.loc[pd.IndexSlice[:, :], bad_idxs_1]
    bad_score_df_1.to_parquet(score_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"), compression="gzip")
    print(bad_score_df_1)
    bad_score_df_2 = score_df.loc[pd.IndexSlice[:, :], bad_idxs_2]
    bad_score_df_2.to_parquet(score_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"), compression="gzip")
    print(bad_score_df_2)
    good_score_df = score_df.loc[pd.IndexSlice[:, :], good_idxs]
    good_score_df.to_parquet(score_data_path.replace(".parquet.gzip", "_good.parquet.gzip"), compression="gzip")
    print(good_score_df)
    print("Done Storing Score Errors\n")


if __name__ == "__main__":
    store_score_and_feature()
