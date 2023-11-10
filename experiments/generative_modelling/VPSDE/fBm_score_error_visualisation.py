import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment
from utils.experiment_evaluations import run_fBm_score_error_experiment
from utils.plotting_functions import plot_score_errors_ts, plot_score_errors_heatmap


def run(config: ConfigDict):
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)

    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path))

    score_errors = run_fBm_score_error_experiment(dataSize=5000, diffusion=diffusion, scoreModel=scoreModel,
                                                        rng=rng,
                                                        config=config)
    start_index = int(0. * config.max_diff_steps)
    end_index = int(1. * config.max_diff_steps)
    time_dim_score_errors = score_errors.mean(axis=1).reshape((config.max_diff_steps, 1))
    plot_score_errors_ts(
        np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)[start_index:end_index],
        time_dim_score_errors[start_index:end_index],
        plot_title="L2 Score Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst, config.timeDim))
    end_index = int(.1 * config.max_diff_steps)
    plot_score_errors_heatmap(score_errors[start_index:end_index, :],
                              plot_title="L2 Score Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                 config.timeDim))


if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.fBm_T2_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)

    run(config)
