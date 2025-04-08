import pickle

import numpy as np
import torch

from src.classes.ClassConditionalStbleTgtLSTMPostMeanDiffTrainer import ConditionalStbleTgtLSTMPostMeanDiffTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_Lorenz96

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.LSTM_4DLorenz.recursive_LSTM_PostMeanScore_4DLorenz_T256_H05_tl_110data_StbleTgt import \
        get_config

    config = get_config()
    assert (config.hurst == 0.5)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 110)
    assert (config.ndims == 4)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalLSTMTSPostMeanScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    #init_experiment(config=config)
    end_epoch = max(config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(end_epoch)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(
            max(1000, min(int(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad) / (
                        config.ts_length - 1)), 1200000)))
        print(training_size)
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_Lorenz96(config=config,H=config.hurst, T=config.ts_length, S=training_size, isUnitInterval=config.isUnitInterval,
                                     initial_state=config.initState,
                                     forcing_const=config.forcing_const,
                                     diff=config.diffusion, ndims=config.ndims)
            np.save(config.data_path, data)
        data = np.concatenate([data[:, [0], :] - np.array(config.initState).reshape((1, 1, config.ndims)), np.diff(data, axis=1)], axis=1)
        data = np.atleast_3d(data[:training_size, :,:])
        assert (data.shape == (training_size, config.ts_length, config.ts_dims))
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                 trainClass=ConditionalStbleTgtLSTMPostMeanDiffTrainer)
    cleanup_experiment()
