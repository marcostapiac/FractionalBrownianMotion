import pickle

import numpy as np
import torch

from src.classes.ClassConditionalStbleTgtMarkovianPostMeanDiffTrainer import \
    ConditionalStbleTgtMarkovianPostMeanDiffTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNewConditionalMarkovianTSPostMeanScoreMatching import \
    NewConditionalMarkovianTSPostMeanScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_fBiPot

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.Markovian_fBiPotDDims.recursive_Markovian_PostMeanScore_fBiPot8Dims_T256_H05_tl_110data_StbleTgt_New import \
        get_config

    config = get_config()
    assert (config.hurst == 0.5)
    assert (config.ndims == 8)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 110)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = NewConditionalMarkovianTSPostMeanScoreMatching(*config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    print(
        config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad) / (config.ts_length - 1))
    init_experiment(config=config)
    end_epoch = max(config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(end_epoch)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(
            max(1000, min(int(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad) / (
                    config.ts_length - 1)), 1200000)))
        training_size -= (training_size % config.ref_batch_size)
        training_size = 10240
        print(training_size)
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fBiPot(num_dims=config.ndims, config=config, T=config.ts_length, isUnitInterval=config.isUnitInterval,
                                   S=training_size,
                                   H=config.hurst, a=config.quartic_coeff, b=config.quad_coeff, c=config.const,
                                   diff=config.diffusion,
                                   initial_state=config.initState)
            np.save(config.data_path, data)
        data = np.concatenate([data[:, [0],:] - config.initState, np.diff(data, axis=1)], axis=1)
        data = np.atleast_3d(data[:training_size, :])
        assert (data.shape == (training_size, config.ts_length, config.ts_dims))
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                 trainClass=ConditionalStbleTgtMarkovianPostMeanDiffTrainer)
    cleanup_experiment()
