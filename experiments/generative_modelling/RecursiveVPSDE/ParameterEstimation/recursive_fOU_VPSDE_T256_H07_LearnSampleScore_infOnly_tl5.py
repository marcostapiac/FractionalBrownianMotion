import pickle

import numpy as np
import pandas as pd
import torch

from src.classes.ClassConditionalLearnSamplerLSTMDiffTrainer import ConditionalLSTMSampleDiffusionModelTrainer
from src.generative_modelling.data_processing import recursive_LSTM_reverse_sampling, \
    train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSSampleScoreMatching import \
    ConditionalLSTMTSSampleScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import cleanup_experiment, init_experiment
from utils.math_functions import generate_fOU

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_LearnSampleScore_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalLSTMTSSampleScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    init_experiment(config=config)
    es = []
    for train_epoch in config.max_epochs:
        try:
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
            final_paths, cond_means, cond_vars = recursive_LSTM_reverse_sampling(diffusion=diffusion,
                                                                                 scoreModel=scoreModel,
                                                                                 data_shape=(
                                                                                     config.dataSize, config.ts_length,
                                                                                     1),
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
            PT = 0 if config.param_time == config.max_diff_steps - 1 else 1
            path_df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip")
            mean_df.to_csv(
                (config.experiment_path + "_NEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU", "fOUm"),
                compression="gzip")
            var_df.to_csv(
                (config.experiment_path + "_NEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU", "fOUv"),
                compression="gzip")
        except FileNotFoundError as e:
            print(e)
            es.append(e)
            continue
    for e in es:
        raise RuntimeError(es)
