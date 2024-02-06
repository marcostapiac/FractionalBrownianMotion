import pickle

import numpy as np
import pandas as pd
import torch

from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model, \
    recursive_LSTM_reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import generate_fBn

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert(config.tdata_mult == 10)

    rng = np.random.default_rng()
    scoreModel = ConditionalTimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)

    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(min(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 1200000))
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fBn(T=config.timeDim, isUnitInterval=config.isUnitInterval, S=training_size, H=config.hurst)
            np.save(config.data_path, data)
        if config.isfBm:
            data = data.cumsum(axis=1)[:training_size, :]
        else:
            data = data[:training_size, :]
        data = np.atleast_3d(data)
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))

    cleanup_experiment()

    final_paths = recursive_LSTM_reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                                  data_shape=(config.dataSize, config.timeDim, 1), config=config)
    df = pd.DataFrame(final_paths)
    df.index = pd.MultiIndex.from_product(
        [["Final Time Samples"], [i for i in range(config.dataSize)]])
    df.to_csv(config.experiment_path + "_Nepochs{}.csv.gzip".format(
        config.max_epochs), compression="gzip")