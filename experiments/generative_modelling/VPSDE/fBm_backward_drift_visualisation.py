import numpy as np
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment
from utils.experiment_evaluations import run_fBm_score_error_experiment, run_fBm_backward_drift_experiment
from utils.plotting_functions import plot_errors_ts, plot_errors_heatmap
from tqdm import tqdm


def run(config: ConfigDict):
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_Nepochs" + str(
                                                                                                            config.max_epochs)))

    drift_errors, score_only_drift_errors = run_fBm_backward_drift_experiment(dataSize=5000, diffusion=diffusion,
                                                                              scoreModel=scoreModel,
                                                                              rng=rng,
                                                                              config=config)
    start_index = int(0. * config.max_diff_steps)
    end_index = int(1. * config.max_diff_steps)

    drift_pic_path = project_config.ROOT_DIR + "experiments/results/drift_plots/DriftErrorTS_fBm_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}_Nepochs{}".format(
        config.hurst, config.timeDim, config.max_diff_steps, config.end_diff_time, config.beta_max,
        config.beta_min, config.max_epochs).replace(
        ".", "")
    score_only_drift_pic_path = drift_pic_path.replace("DriftErrorsTS", "ScoreOnlyDriftErrorsTS")

    time_dim_drift_errors = drift_errors.mean(axis=1).reshape((config.max_diff_steps, 1))
    time_dim_score_only_drift_errors = score_only_drift_errors.mean(axis=1).reshape((config.max_diff_steps, 1))

    plot_errors_ts(
        np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)[start_index:end_index],
        time_dim_drift_errors[start_index:end_index],
        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst, config.timeDim),
        path=drift_pic_path)

    plot_errors_ts(
        np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)[start_index:end_index],
        time_dim_score_only_drift_errors[start_index:end_index],
        plot_title="MSE Score Only Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                             config.timeDim),
        path=score_only_drift_pic_path)

    drift_hm_path = drift_pic_path.replace("DriftErrorTS", "DriftErrorHM")
    score_only_drift_hm_path = drift_hm_path.replace("DriftErrorHM", "ScoreOnlyDriftErrorHM")

    start_index = int(0. * config.max_diff_steps)
    end_index = int(.05 * config.max_diff_steps)
    plot_errors_heatmap(drift_errors[start_index:end_index, :],
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                        config.timeDim),
                        path=drift_hm_path)

    plot_errors_heatmap(drift_errors[start_index:end_index, :],
                        plot_title="MSE Score Only Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(
                                  config.hurst,
                                  config.timeDim),
                        path=score_only_drift_hm_path)


if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.fBm_T32_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)

    run(config)
