import os

import numpy as np
import torch
from tqdm import tqdm

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fBiPot.recursive_LSTM_PostMeanScore_fBiPot_T256_H05_tl_110data import \
    get_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.drift_evaluation_functions import multivar_score_based_LSTM_drift_OOS


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -(4. * config.quartic_coeff * np.power(prev, 3) + 2. * config.quad_coeff * prev + config.const)
    return drift_X[:, np.newaxis, :]


if __name__ == "__main__":
    config = get_config()
    assert ("BiPot" in config.data_path)

    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")

    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    for Nepoch in config.max_epochs:
        print(f"Starting Epoch {Nepoch}\n")
        num_diff_times = 1
        rmse_quantile_nums = 20
        num_paths = 100
        num_time_steps = 100
        all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        all_global_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
        for quant_idx in tqdm(range(rmse_quantile_nums)):
            PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters)
            PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
            PM = PM.to(device)

            num_paths = 100
            num_time_steps = 100
            deltaT = config.deltaT
            initial_state = np.repeat(np.atleast_2d(config.initState)[np.newaxis, :], num_paths, axis=0)
            print(initial_state.shape)
            assert (initial_state.shape == (num_paths, 1, config.ndims))

            true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            global_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
            local_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

            # Initialise the "true paths"
            true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
            # Initialise the "global score-based drift paths"
            global_states[:, [0], :] = true_states[:, [0], :]
            local_states[:, [0], :] = true_states[:, [0],
                                      :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)

            # Euler-Maruyama Scheme for Tracking Errors
            global_h, global_c = None, None
            local_h, local_c = None, None
            for i in tqdm(range(1, num_time_steps + 1)):
                eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)
                assert (eps.shape == (num_paths, 1, config.ndims))
                true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config)

                true_states[:, [i], :] = true_states[:, [i - 1], :] \
                                         + true_drift(true_states[:, i - 1, :], num_paths=num_paths,
                                                      config=config) * deltaT \
                                         + eps
                global_mean, global_h, global_c = multivar_score_based_LSTM_drift_OOS(score_model=PM, time_idx=i - 1,
                                                                                      h=global_h, c=global_c,
                                                                                      num_diff_times=num_diff_times,
                                                                                      diffusion=diffusion,
                                                                                      num_paths=num_paths,
                                                                                      ts_step=deltaT, config=config,
                                                                                      device=device,
                                                                                      prev=global_states[:, i - 1, :])

                global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
                local_mean, local_h, local_c = multivar_score_based_LSTM_drift_OOS(score_model=PM, time_idx=i - 1,
                                                                                   h=local_h, c=local_c,
                                                                                   num_diff_times=num_diff_times,
                                                                                   diffusion=diffusion,
                                                                                   num_paths=num_paths, ts_step=deltaT,
                                                                                   config=config,
                                                                                   device=device,
                                                                                   prev=true_states[:, i - 1, :])

                local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
            all_true_states[quant_idx, :, :, :] = true_states
            all_global_states[quant_idx, :, :, :] = global_states
            all_local_states[quant_idx, :, :, :] = local_states
        if "_ST_" in config.scoreNet_trained_path:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_ST_fBiPot_OOSDriftTrack_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                ".", "")
        else:
            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fBiPot_OOSDriftTrack_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                ".", "")
        print(f"Save path:{save_path}\n")
        np.save(save_path + "_global_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)
