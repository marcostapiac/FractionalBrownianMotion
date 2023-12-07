import numpy as np
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import run_fBm_score
from utils.plotting_functions import make_gif


def run(config: ConfigDict) -> None:
    try:
        assert (config.save_freq <= config.max_diff_steps)
        assert (config.save_freq > 0)
        assert (config.dim1 < config.timeDim)
        assert (config.dim2 < config.timeDim)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    rng = np.random.default_rng()
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    dim_pair = torch.Tensor([config.dim1, config.dim2]).to(torch.int32)

    try:
        scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
            *config.model_parameters)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_Nepochs" + str(
                                                                                                            config.max_epochs)))

    folder_path = project_config.ROOT_DIR + "experiments/results/backward_gifs/"

    gif_path = "fBm_dimPair{}_dimPair{}_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}_Nepochs{}".format(
        dim_pair[0],
        dim_pair[1],
        config.hurst,
        config.timeDim,
        config.max_diff_steps,
        config.end_diff_time,
        config.beta_max,
        config.beta_min, config.max_epochs).replace(
        ".", "")
    run_fBm_score(dataSize=config.dataSize, dim_pair=dim_pair, diffusion=diffusion, scoreModel=scoreModel, rng=rng,
                  config=config, folderPath=folder_path, gifPath=gif_path)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.fBm_T32_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)
    assert(config.early_stop_idx == 0)
    config.gif_save_freq = int((0.05 * config.max_diff_steps))
    config.dataSize = 5000
    config.dim1 = 0
    config.dim2 = 1
    # Run experiments
    run(config)
    if config.timeDim == 32:
        config.dim1 = 30
        config.dim2 = 31
        run(config)
    elif config.timeDim == 256:
        config.dim1 = 254
        config.dim2 = 255
        run(config)
