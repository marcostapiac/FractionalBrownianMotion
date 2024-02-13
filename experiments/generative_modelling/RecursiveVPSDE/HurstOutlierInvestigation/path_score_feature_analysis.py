import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.plotting_functions import hurst_estimation

def identify_jump_index(time_series, eps):
    # Calculate the differences between consecutive values
    differences = np.diff(time_series)

    # Find the index where the first large jump occurs
    jump_index = np.argmax(np.abs(differences) > eps)

    # Identify the index just before the jump
    if jump_index > 0:
        index_before_jump = jump_index - 1
        return index_before_jump
    else:
        return None  # No large jump found

def path_score_feature_analysis() -> None:
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config
    config = get_config()

    # Now plot Hurst histogram for the generated samples
    bad_path_idxs = []
    bad_path_times = [] # Times just before a large jump occured
    for train_epoch in [1920]:  # config.max_epochs:
        path_df = pd.read_csv(config.experiment_path + "_Nepochs{}.csv.gzip".format(train_epoch), compression="gzip",
                         index_col=[0, 1]).iloc[4000:4100,:]
        path_df = path_df.apply(
            lambda x: [eval(i.replace("(", "").replace(")", "").replace("tensor", "")) if type(i) == str else i for i in
                       x]).loc["Final Time Samples"]
        hs = hurst_estimation(path_df.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst)
        hs.index = path_df.index
        lsp = np.linspace(1, path_df.shape[1] + 1, path_df.shape[1])
        # Under-estimation
        bad_idxs = hs.index[hs.lt(0.7).any(axis=1)] # Path IDs which satisfy condition
        if not bad_idxs.empty:
            bad_path_idxs.append(bad_idxs)
            bad_paths = path_df.loc[bad_idxs,:]
            bad_time = identify_jump_index(time_series=bad_paths, eps=1.1)
            if bad_time is not None: bad_path_times.append(bad_time)
            for idx in bad_idxs:
                path = bad_paths.loc[idx, :]
                plt.plot(lsp, path, label=(idx, round(hs.loc[idx][0], 2)))
            plt.title("fBm Bad Hurst Paths")
            plt.legend()
            plt.xlabel("Time")
            plt.show()
        # Over estimation
        bad_idxs = hs.index[hs.gt(0.8).any(axis=1)]
        if not bad_idxs.empty:
            bad_path_idxs.append(bad_idxs)
            bad_paths = path_df.loc[bad_idxs, :]
            bad_time = identify_jump_index(time_series=bad_paths, eps=1.1)
            if bad_time is not None: bad_path_times.append(bad_time)
            for idx in bad_idxs:
                path = bad_paths.loc[idx, :]
                plt.plot(lsp, path, label=(idx, round(hs.loc[idx][0], 2)))
            plt.title("fBm Bad Hurst Paths")
            plt.legend()
            plt.xlabel("Time")
            plt.show()
        # Now proceed to plot the drift score for the bad path indices
        drift_error_df = pd.read_csv(config.experiment_path.replace("results/", "results/drift_data/") + "_Nepochs{}_SFS".format(train_epoch).replace(".", "") + ".parquet.gzip", compression="gzip", index_col = [0,1], names=bad_path_idxs, engine="pyarrow")
        for bad_idxs in bad_path_idxs:
            for idx in range(len(bad_idxs)):
                drift_error_path = drift_error_df.iloc[pd.IndexSlice[bad_path_times[idx]+1, :] [idx]]
                plt.plot(lsp, drift_error_path, label=(bad_paths.index[idx], round(hs.iloc[bad_paths.index[idx], 0], 2)))
            plt.title("Drift Error for Bad Paths")
            plt.legend()
            plt.xlabel("Diffusion Time")
            plt.show()
            plt.close()
        # Now proceed to plot the feature time series for those paths
        # TODO: How to visualise 40 dimensional feature vector over whole real-time
        feature_df = pd.read_csv(config.experiment_path.replace("results/", "results/feature_data/") + "_Nepochs{}_SFS".format(train_epoch).replace(".", "") + ".parquet.gzip", compression="gzip", index_col = [0,1])
        for bad_idxs in bad_path_idxs:
            for idx in range(len(bad_idxs)):
                feature_path = feature_df.iloc[:, [idx], :]
                plt.plot(lsp, feature_path, label=(bad_paths.index[idx], round(hs.iloc[bad_paths.index[idx], 0], 2)))
                plt.title("fBm feature path for bad Hurst")
                plt.legend()
                plt.xlabel("Time")
                plt.show()
                print(bad_paths)
if __name__ == "__main__":
    path_score_feature_analysis()
