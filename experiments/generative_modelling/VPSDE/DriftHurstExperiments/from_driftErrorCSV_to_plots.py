import numpy as np
import pandas as pd

from configs import project_config
from utils.plotting_functions import plot_errors_heatmap, plot_errors_ts

if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()

    drift_data_path = project_config.ROOT_DIR + "experiments/results/drift_data/DriftErrorsTS_fBm_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}_Nepochs{}".format(
        config.hurst, config.timeDim, config.max_diff_steps, config.end_diff_time, config.beta_max,
        config.beta_min, config.max_epochs).replace(
        ".", "")
    drift_pic_path = drift_data_path.replace("/drift_data/", "/drift_plots/")

    time_space = np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)

    start_index = 0
    end_index = 10 #config.max_diff_steps//2
    time_idxs = [i for i in range(start_index, end_index)]
    drift_errors = pd.read_csv(drift_data_path + ".csv.gzip", compression="gzip", index_col=[0])
    drift_hm_path = drift_pic_path.replace("DriftErrorsTS", "DriftErrorsHM")
    time_dim_drift_errors = drift_errors.mean(axis=1).to_numpy().reshape((config.max_diff_steps, 1))
    plot_errors_ts(
        time_space[time_idxs],
        time_dim_drift_errors[time_idxs],
        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst, config.timeDim),
        path=drift_pic_path)

    # score_only_drift_data_path = drift_data_path.replace("DriftErrorsTS", "ScoreOnlyDriftErrorsTS")
    # score_only_drift_errors = pd.read_csv(score_only_drift_data_path + ".csv.gzip", compression="gzip", index_col=[0])
    #score_only_drift_pic_path = score_only_drift_data_path.replace("/drift_data/", "/drift_plots/")
    score_only_drift_hm_path = drift_hm_path.replace("DriftErrorsHM", "ScoreOnlyDriftErrorsHM")
    # time_dim_score_only_drift_errors = score_only_drift_errors.mean(axis=1).to_numpy().reshape(
    #    (config.max_diff_steps, 1))
    # plot_errors_ts(
    #    np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)[start_index:end_index],
    #    time_dim_score_only_drift_errors[start_index:end_index],
    #    plot_title="MSE Score Only Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
    #                                                                                        config.timeDim),
    #    path=score_only_drift_pic_path)

    start_index = 0
    end_index = 10

    time_idxs = [i for i in range(start_index, end_index+1)]
    dims = [i for i in range(0, config.timeDim)]
    plot_errors_heatmap(drift_errors.iloc[time_idxs, dims].to_numpy(),
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                  config.timeDim),
                        path=drift_hm_path,xticks=dims, yticks=time_idxs)

    dims = [i for i in range(0, config.timeDim)][::2]
    plot_errors_heatmap(drift_errors.iloc[time_idxs, dims].to_numpy(),
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                  config.timeDim),
                        path=drift_hm_path, xticks=dims, yticks=time_idxs)

    dims = [i for i in range(0, config.timeDim)][1::2]
    plot_errors_heatmap(drift_errors.iloc[time_idxs, dims].to_numpy(),
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(
                            config.hurst,
                            config.timeDim),
                        path=score_only_drift_hm_path, xticks=dims, yticks=time_idxs)

    dims = [i for i in range(0, config.timeDim)][1::2][:int(256 / 2) - 1]
    plot_errors_heatmap(drift_errors.iloc[time_idxs, dims].to_numpy(),
                        plot_title="MSE Drift Error for VPSDE fBm with $(H, T) = ({},{})$".format(
                            config.hurst,
                            config.timeDim),
                        path=score_only_drift_hm_path, xticks=dims, yticks=time_idxs)
