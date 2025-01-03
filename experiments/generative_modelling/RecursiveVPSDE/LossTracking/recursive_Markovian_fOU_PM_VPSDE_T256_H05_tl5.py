import pickle

import numpy as np
import torch

from src.classes.ClassConditionalPostMeanMarkovianDiffTrainer import ConditionalPostMeanMarkovianDiffTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import cleanup_experiment, init_experiment
from utils.math_functions import generate_fOU

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.Markovian_fOU.recursive_Markovian_PostMeanScore_fOU_T256_H05_tl_5data import get_config

    config = get_config()
    assert (config.hurst == 0.5)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    assert (config.weightings == True)
    print(config.scoreNet_trained_path, config.dataSize)
    scoreModel = ConditionalMarkovianTSPostMeanScoreMatching(
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
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                 trainClass=ConditionalPostMeanMarkovianDiffTrainer)
    cleanup_experiment()
