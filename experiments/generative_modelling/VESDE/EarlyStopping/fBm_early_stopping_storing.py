import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


def run_early_stopping(config: ConfigDict) -> None:
    T = config.timeDim

    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))

    synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                     scoreModel=scoreModel,
                                     config=config).cpu().numpy().reshape((config.dataSize, T))
    config.early_stop_idx = 0
    no_stop_synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                             scoreModel=scoreModel,
                                             config=config).cpu().numpy().reshape((config.dataSize, T))
    df1 = pd.DataFrame(synth_samples)
    df2 = pd.DataFrame(no_stop_synth_samples)
    df = pd.concat([df1, df2], ignore_index=False)
    df.index = pd.MultiIndex.from_product(
        [["Early Stop Synthetic", "No Early Stop Synthetic"], [i for i in range(config.dataSize)]])
    df.to_csv(config.experiment_path.replace("/results/", "/results/early_stopping/") + "_Samples_EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(config.max_epochs), compression="gzip")

if __name__ == "__main__":
    from configs.VESDE.fBm_T32_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1)
    assert (config.early_stop_idx == 1)
    config.dataSize = 5000
    run_early_stopping(config)