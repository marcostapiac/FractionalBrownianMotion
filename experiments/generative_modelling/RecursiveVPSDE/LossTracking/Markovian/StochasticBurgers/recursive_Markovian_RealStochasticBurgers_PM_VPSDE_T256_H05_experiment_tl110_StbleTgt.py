import pickle

import numpy as np
import torch

from src.classes.ClassConditionalStbleTgtMarkovianPostMeanDiffTrainer import \
    ConditionalStbleTgtMarkovianPostMeanDiffTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_StochasticBurgers
from utils.resource_logger import ResourceLogger

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.Markovian_StochasticBurgers.recursive_Markovian_PostMeanScore_RealStochasticBurgers_T256_H05_tl_110data_StbleTgt import \
        get_config
    config = get_config()
    with ResourceLogger(
            interval=120,
            outfile=config.resource_logging_path,  # path where log will be written
            job_type="GPU training",
    ):
        assert (config.hurst == 0.5)
        assert config.real
        assert (config.early_stop_idx == 0)
        assert (config.tdata_mult == 110)
        assert (config.feat_thresh == 1./500.)
        print(config.scoreNet_trained_path, config.dataSize)
        rng = np.random.default_rng()
        scoreModel = ConditionalMarkovianTSPostMeanScoreMatching(
            *config.model_parameters)
        diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
        print(
            config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad) / (config.ts_length - 1))
        #init_experiment(config=config)
        end_epoch = max(config.max_epochs)
        try:
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(end_epoch)))
        except FileNotFoundError as e:
            print("Error {}; no valid trained model found; proceeding to training\n".format(e))
            training_size = int(
                max(1000, min(int(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad) / (
                            config.ts_length - 1)), 10240)))
            training_size -= (training_size % config.ref_batch_size)
            training_size = 10240
            print(training_size)
            try:
                data = np.load(config.data_path, allow_pickle=True)
                assert (data.shape[0] >= training_size)
            except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
                print("Error {}; generating synthetic data\n".format(e))
                data = generate_StochasticBurgers(config=config, T=config.ts_length, isUnitInterval=config.isUnitInterval,
                                         S=training_size,
                                         H=config.hurst, nu=config.nu, num_fourier_modes=config.num_fourier_modes, alpha=config.alpha,ndims=config.num_dims,
                                         diff=config.diffusion,
                                         initial_state=config.initState)
                np.save(config.data_path, data)
            data = data[:, :, :, 0] if config.real else data[:, :, :, 1]
            data = np.concatenate([data[:, [0],:] - config.initState, np.diff(data, axis=1)], axis=1)
            data = np.atleast_3d(data[:training_size, :])
            print(data.shape, (training_size, config.ts_length, config.ts_dims))
            assert (data.shape == (training_size, config.ts_length, config.ts_dims))
            # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
            train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel,
                                                     trainClass=ConditionalStbleTgtMarkovianPostMeanDiffTrainer)
        cleanup_experiment()
