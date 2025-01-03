import pickle

import numpy as np
import torch

from src.classes.ClassConditionalMarkovianWithPositionDiffTrainer import \
    ConditionalMarkovianWithPositionDiffusionModelTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import \
    ConditionalMarkovianTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_3DLorenz

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.Markovian_3DLorenz.recursive_Markovian_3DLorenzWithPosition_T256_H05_tl_5data import \
        get_config

    config = get_config()
    assert (config.hurst == 0.5)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalMarkovianTSScoreMatching(
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
            data = generate_3DLorenz(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                                     H=config.hurst, beta=config.ts_beta, rho=config.ts_rho, sigma=config.ts_sigma,
                                     diff=config.ts_diff,
                                     initial_state=config.initState)
            np.save(config.data_path, data)
        data = np.concatenate([data[:, [0]], np.diff(data, axis=1)], axis=1)
        data = np.atleast_3d(data[:training_size, :])
        assert (data.shape == (training_size, config.ts_length, config.ts_dims))
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                 trainClass=ConditionalMarkovianWithPositionDiffusionModelTrainer)
    cleanup_experiment()
