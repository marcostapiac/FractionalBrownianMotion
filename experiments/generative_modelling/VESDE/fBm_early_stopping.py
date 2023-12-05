import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.math_functions import optimise_whittle, reduce_to_fBn


def run_early_stopping(config: ConfigDict) -> None:
    rng = np.random.default_rng()
    H = config.hurst
    T = config.timeDim

    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    fbn = FractionalBrownianNoise(H=H, rng=rng)

    exact_samples = np.zeros((config.dataSize, T))
    for j in tqdm(range(config.dataSize), desc="Exact Sampling ::", dynamic_ncols=False, position=0):
        exact_samples[j, :] = fbn.circulant_simulation(N_samples=T).cumsum()

    synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                     scoreModel=scoreModel,
                                     config=config).cpu().numpy().reshape((config.dataSize, T))
    config.early_stop_idx = 0
    no_stop_synth_samples = reverse_sampling(data_shape=(config.dataSize, config.timeDim), diffusion=diffusion,
                                             scoreModel=scoreModel,
                                             config=config).cpu().numpy().reshape((config.dataSize, T))
    samples_dict = {"Synthetic": synth_samples,"No Early Stop Synthetic": no_stop_synth_samples}
    df = pd.DataFrame.from_dict(samples_dict)
    df.to_csv(config.experiment_path + "_Samples_EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(config.max_epochs))

    true_Hs = []
    synth_Hs = []
    no_stop_synth_Hs = []
    approx_true_fBn = reduce_to_fBn(exact_samples, reduce=True)
    approx_fBn = reduce_to_fBn(synth_samples, reduce=True)
    approx_no_stop_fBn = reduce_to_fBn(no_stop_synth_samples, reduce=True)
    for j in tqdm(range(config.dataSize), desc="Estimating Hurst Parameters ::", dynamic_ncols=False, position=0):
        ht = optimise_whittle(data=approx_true_fBn, idx=j)
        h = optimise_whittle(data=approx_fBn, idx=j)
        nh = optimise_whittle(data=approx_no_stop_fBn, idx=j)
        true_Hs.append(ht)
        synth_Hs.append(h)
        no_stop_synth_Hs.append(nh)

    keys = ["True Hs", "Synthetic Hs", "No Early Stop Hs"]
    results_dict = {keys[0]: true_Hs, keys[1]: synth_Hs, keys[2]: no_stop_synth_Hs}
    df = pd.DataFrame.from_dict(data=results_dict)
    df.to_csv(config.experiment_path + "_EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(config.max_epochs),
              compression="gzip", index=True)


if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1)
    assert (config.early_stop_idx == 1)
    config.dataSize = 5000
    run_early_stopping(config)
