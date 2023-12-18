import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


def run_early_stopping(config: ConfigDict) -> None:
    T = config.timeDim

    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))

    config.early_stop_idx = 1
    synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                     scoreModel=scoreModel,
                                     config=config).cpu().numpy().reshape((config.dataSize, T))
    early_stop_idx = config.early_stop_idx

    config.early_stop_idx = 0
    no_stop_synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                             scoreModel=scoreModel,
                                             config=config).cpu().numpy().reshape((config.dataSize, T))
    df1 = pd.DataFrame(synth_samples)
    df2 = pd.DataFrame(no_stop_synth_samples)
    df = pd.concat([df1, df2], ignore_index=False)
    df.index = pd.MultiIndex.from_product(
        [["Early Stop {}".format(early_stop_idx), "Final Time Samples"], [i for i in range(config.dataSize)]])
    df.to_csv(config.experiment_path.replace("/results/",
                                             "/results/early_stopping/") + "_Samples_EStop{}_Nepochs{}.csv.gzip".format(
        early_stop_idx, config.max_epochs), compression="gzip")


if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1)
    assert (config.early_stop_idx == 0)
    config.dataSize = 10000
    run_early_stopping(config)
