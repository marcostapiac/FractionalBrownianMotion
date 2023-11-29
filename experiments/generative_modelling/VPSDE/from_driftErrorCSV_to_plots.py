import numpy as np
import pandas as pd

from configs import project_config
from utils.plotting_functions import plot_errors_heatmap, plot_errors_ts

if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()

    drift_pic_path = project_config.ROOT_DIR + "experiments/results/drift_plots/DriftErrorsTS_fBm_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}_Nepochs{}".format(
        config.hurst, config.timeDim, config.max_diff_steps, config.end_diff_time, config.beta_max,
        config.beta_min, config.max_epochs).replace(
        ".", "")
    score_only_drift_pic_path = drift_pic_path.replace("DriftErrorsTS", "ScoreOnlyDriftErrorsTS")

    start_index = int(0. * config.max_diff_steps)
    end_index = int(min(0.01 * config.max_diff_steps, 20))

    drift_errors = pd.read_csv(drift_pic_path + ".csv.gzip", compression="gzip", index_col=[0])
    score_only_drift_errors = pd.read_csv(score_only_drift_pic_path + ".csv.gzip", compression="gzip", index_col=[0])
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

    drift_hm_path = drift_pic_path.replace("DriftErrorsTS", "DriftErrorsHM")
    score_only_drift_hm_path = drift_hm_path.replace("DriftErrorsHM", "ScoreOnlyDriftErrorsHM")

    start_index = int(0. * config.max_diff_steps)
    end_index = int(min(0.01 * config.max_diff_steps, 20))

    plot_errors_heatmap(drift_errors[start_index:end_index, :],
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                  config.timeDim),
                        path=drift_hm_path)

    plot_errors_heatmap(drift_errors[start_index:end_index, :],
                        plot_title="MSE Score Only Drift Error for VESDE fBm with $(H, T) = ({},{})$".format(
                            config.hurst,
                            config.timeDim),
                        path=score_only_drift_hm_path)
