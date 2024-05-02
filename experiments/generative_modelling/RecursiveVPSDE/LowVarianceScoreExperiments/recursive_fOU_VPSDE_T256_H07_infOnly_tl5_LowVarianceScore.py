import numpy as np
import pandas as pd
import torch

from src.generative_modelling.data_processing import recursive_LSTM_reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSScoreMatching import \
    ConditionalLSTMTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalLSTMTSScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    init_experiment(config=config)
    es = []
    param_times = [config.max_diff_steps - 1]
    train_epoch = 2920
    config.predictor_model = "CondReverseDiffusion"
    for param_time in param_times:
        try:
            config.param_time = param_time
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
            final_paths, cond_means, cond_vars = recursive_LSTM_reverse_sampling(diffusion=diffusion,
                                                                                 scoreModel=scoreModel,
                                                                                 data_shape=(
                                                                                 config.dataSize, config.ts_length, 1),
                                                                                 config=config)
            path_df = pd.DataFrame(final_paths)
            path_df.index = pd.MultiIndex.from_product(
                [["Final Time Samples"], [i for i in range(config.dataSize)]])
            mean_df = pd.DataFrame(cond_means)
            mean_df.index = pd.MultiIndex.from_product(
                [["Final Time Means"], [i for i in range(config.dataSize)]])
            var_df = pd.DataFrame(cond_vars)
            var_df.index = pd.MultiIndex.from_product(
                [["Final Time Vars"], [i for i in range(config.dataSize)]])
            if config.param_time == config.max_diff_steps - 1:
                PT = 0
            elif config.param_time == 4600:
                PT = 2
            else:
                PT = 1
            path_df.to_csv(config.experiment_path + "_rdNEp{}.csv.gzip".format(train_epoch), compression="gzip")
            mean_df.to_csv((config.experiment_path + "_rdNEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU",
                                                                                                               "fOUm").replace(
                "fOUm00", "fm00"), compression="gzip")
            var_df.to_csv((config.experiment_path + "_rdNEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU",
                                                                                                              "fOUv").replace(
                "fOUv00", "fv00"), compression="gzip")
        except FileNotFoundError as e:
            print(e)
            es.append(e)
            continue
    for e in es:
        raise RuntimeError(e)
