import numpy as np
import pandas as pd

from utils.plotting_functions import plot_errors_heatmap, plot_errors_ts

if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()

    drift_data_path = config.experiment_path.replace("results/",
                                                     "results/drift_data/") + "_DriftErrorsTS_NEp{}".format(
        config.max_epochs).replace(
        ".", "")

    drift_pic_path = drift_data_path.replace("/drift_data/", "/drift_plots/")
    time_space = np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)
    start_index = 1  # int(0.0 * config.max_diff_steps)
    end_index = int(.1 * config.max_diff_steps)

    time_idxs = [i for i in range(start_index, end_index)]
    drift_errors = pd.read_csv(drift_data_path + ".csv.gzip", compression="gzip", index_col=[0])
    drift_hm_path = drift_pic_path.replace("DriftErrorsTS", "DriftErrorsHM")
    dims = [i for i in range(config.timeDim)]
    time_dim_drift_errors = drift_errors.iloc[:, dims]
    time_dim_drift_errors = time_dim_drift_errors.mean(axis=1).to_numpy().reshape((config.max_diff_steps, 1))

    plot_errors_ts(
        time_space[time_idxs],
        time_dim_drift_errors[time_idxs],
        plot_title="MSE Drift Error for VESDE fBm with $(H, T) = ({},{})$".format(config.hurst, config.timeDim),
        path=drift_pic_path)
    time_dim_drift_errors = (time_dim_drift_errors[::-1].cumsum() / np.arange(1, config.max_diff_steps + 1))[::-1]

    plot_errors_ts(
        time_space[time_idxs],
        time_dim_drift_errors[time_idxs],
        plot_title="MSE Drift CumMean Error for VESDE fBm with $(H, T) = ({},{})$".format(config.hurst, config.timeDim),
        path=drift_pic_path)

    start_index = 0
    end_index = 10

    time_idxs = [i for i in range(start_index, end_index)]
    dims = [i for i in range(0, config.timeDim)]
    plot_errors_heatmap(drift_errors.iloc[time_idxs, dims].to_numpy(),
                        plot_title="MSE Drift Error for VESDE fBm with $(H, T) = ({},{})$".format(config.hurst,
                                                                                                  config.timeDim),
                        path=drift_hm_path, xticks=dims, yticks=time_idxs)
