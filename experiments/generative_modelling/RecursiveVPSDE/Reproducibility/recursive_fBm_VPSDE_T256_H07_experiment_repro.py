import os
import random

import numpy as np
import pandas as pd
import torch

from src.generative_modelling.data_processing import recursive_LSTM_reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import init_experiment, cleanup_experiment


def repro_weights_init(m):
    if isinstance(m, torch.nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Note: CUDA >=10.2 Need to set CUBLAS_WORKSPACE_CONFIG =:4096:2 in command line before torchrun
    # Note: CUDA <=10.1 Need to set CUDA_LAUNCH_BLOCKING=1 in command line before torchrun
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Data parameters
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert(config.tdata_mult == 10)

    scoreModel = ConditionalTimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.apply(repro_weights_init)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)

    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        assert FileNotFoundError("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(min(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 1200000))
    cleanup_experiment()

    numpaths = 5
    final_paths = recursive_LSTM_reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                               data_shape=(numpaths, config.timeDim, 1), config=config)
    df = pd.DataFrame(final_paths)
    df.index = pd.MultiIndex.from_product(
        [["Final Time Samples"], [i for i in range(numpaths)]])
    df.to_csv(config.experiment_path.replace("/results/", "/results/early_stopping/") + "_Nepochs{}_Mseed{}.csv.gzip".format(
        config.max_epochs, seed), compression="gzip")
