import pickle

import numpy as np
import pandas as pd
import torch

from experiments.generative_modelling.RecursiveVPSDE.ParameterEstimation.PM.TS_single_model_drift_eval import \
    TS_drift_eval
from src.classes.ClassConditionalLSTMDiffTrainer import ConditionalLSTMDiffusionModelTrainer
from src.classes.ClassConditionalLSTMWithPositionDiffTrainer import ConditionalLSTMWithPositionDiffusionModelTrainer
from src.generative_modelling.data_processing import recursive_LSTM_reverse_sampling, \
    train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSScoreMatching import \
    ConditionalLSTMTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_fOU

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_fOU_T256_H05_tl_5data import get_config

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
    end_epoch = max(config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(end_epoch)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(
            min(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 1200000))
        print(training_size)
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fOU(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                                H=config.hurst, mean_rev=config.mean_rev, mean=config.mean, diff=config.diffusion,
                                initial_state=config.initState)
            np.save(config.data_path, data)
        data = np.concatenate([data[:, [0]], np.diff(data, axis=1)], axis=1)
        data = np.atleast_3d(data[:training_size, :])
        assert (data.shape == (training_size, config.ts_length, config.ts_dims))
        print(config.hurst)
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                 trainClass=ConditionalLSTMWithPositionDiffusionModelTrainer)
    cleanup_experiment()
    TS_drift_eval()
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
