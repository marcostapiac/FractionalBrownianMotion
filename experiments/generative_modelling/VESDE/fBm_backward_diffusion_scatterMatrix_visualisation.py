import numpy as np
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import run_fBm_scatter_matrix
from utils.plotting_functions import make_gif


def run(config: ConfigDict) -> None:
    try:
        assert (config.save_freq <= config.max_diff_steps)
        assert (config.save_freq > 0)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    rng = np.random.default_rng()
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)

    try:
        scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
            *config.model_parameters)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_Nepochs" + str(
                                                                                                            config.max_epochs)))

    gif_path = config.experiment_path.replace("experiments/results/", "experiments/results/scatter_matrix_gifs/") + "_r{}_c{}".format(
        config.row_idxs, config.col_idxs).replace(
        ".", "").replace("[", "").replace("]", "").replace(" ", ",")
    run_fBm_scatter_matrix(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, rng=rng,
                           config=config, folderPath="", gifPath=gif_path)
    make_gif("", gif_path)


if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)
    assert(config.early_stop_idx == 0)
    config.dataSize = 10000
    config.idx_start_save = int(0.0 * config.max_diff_steps)
    config.gif_save_freq = int(max(0.05 * (config.max_diff_steps - config.idx_start_save), 1))
    config.row_idxs = np.array([227,245])
    config.col_idxs = np.array([227,245])
    run(config)