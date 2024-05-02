import pandas as pd
import torch
from ml_collections import ConfigDict

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching


def run_early_stopping(config: ConfigDict) -> None:
    T = config.ts_length

    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    scoreModel = TSScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))

    config.early_stop_idx = 393
    synth_samples = reverse_sampling(data_shape=(config.dataSize, config.ts_length), diffusion=diffusion,
                                     scoreModel=scoreModel,
                                     config=config).cpu().numpy().reshape((config.dataSize, T))
    early_stop_idx = config.early_stop_idx

    config.early_stop_idx = 0
    no_stop_synth_samples = reverse_sampling(data_shape=(config.dataSize, config.ts_length), diffusion=diffusion,
                                             scoreModel=scoreModel,
                                             config=config).cpu().numpy().reshape((config.dataSize, T))
    df1 = pd.DataFrame(synth_samples)
    df2 = pd.DataFrame(no_stop_synth_samples)
    df = pd.concat([df1, df2], ignore_index=False)
    df.index = pd.MultiIndex.from_product(
        [["Early Stop {}".format(early_stop_idx), "Final Time Samples"], [i for i in range(config.dataSize)]])
    df.to_csv(config.experiment_path.replace("/results/",
                                             "/results/early_stopping/") + "_EStop{}_NEp{}.csv.gzip".format(
        early_stop_idx, config.max_epochs), compression="gzip")


if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1)
    assert (config.early_stop_idx == 0)
    config.dataSize = 10000
    run_early_stopping(config)
