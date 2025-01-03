import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching
from utils.experiment_evaluations import run_fBm_score_error_experiment
from utils.plotting_functions import plot_errors_ts, plot_errors_heatmap


def run(config: ConfigDict):
    rng = np.random.default_rng()
    scoreModel = TSScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_NEp" + str(
                                                                                                            config.max_epochs)))

    score_errors = run_fBm_score_error_experiment(dataSize=10000, diffusion=diffusion, scoreModel=scoreModel,
                                                  rng=rng,
                                                  config=config)
    increment = "CumSum" if config.isfBm else "Inc"
    unitInterval = "UnitIntv" if config.isUnitInterval else "StdIntv"

    start_index = int(0. * config.max_diff_steps)
    end_index = int(1. * config.max_diff_steps)

    pic_path = config.experiment_path.replace("experiments/results/",
                                              "experiments/results/score_plots/") + "ScoreErrorTS_NEp{}".format(
        config.max_epochs).replace(
        ".", "")

    time_dim_score_errors = score_errors.mean(axis=1).reshape((config.max_diff_steps, 1))
    plot_errors_ts(
        np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)[start_index:end_index],
        time_dim_score_errors[start_index:end_index],
        plot_title="MSE Score VESDE {} {} LSTM_fBm with $(H, T) = ({},{})$".format(increment, unitInterval, config.hurst,
                                                                              config.ts_length),
        path=pic_path)

    pic_path = pic_path.replace("ScoreErrorTS", "ScoreErrorHM")

    start_index = int(0. * config.max_diff_steps)
    end_index = int(.2 * config.max_diff_steps)
    dims = [i for i in range(config.ts_length)]
    times = np.linspace(start_index, end_index)
    plot_errors_heatmap(score_errors[start_index:end_index, :],
                        plot_title="MSE Score VESDE {} {} LSTM_fBm with $(H, T) = ({},{})$".format(increment, unitInterval,
                                                                                              config.hurst,
                                                                                              config.ts_length),
                        path=pic_path, xticks=dims, yticks=list(times))


if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)
    assert (config.early_stop_idx == 0)

    run(config)
