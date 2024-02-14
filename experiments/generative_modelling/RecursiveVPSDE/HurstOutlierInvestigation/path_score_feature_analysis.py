import seaborn as sns
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
    bad_path_times = [] # Times just before a large jump occured
    for train_epoch in [1920]:  # config.max_epochs:
        path_df_path = config.experiment_path + "_Nepochs{}_SFS.parquet.gzip".format(train_epoch)
        path_df = pd.read_parquet(path_df_path, engine="pyarrow")
        drift_data_path = config.experiment_path.replace("results/",
                                                         "results/drift_data/") + "_Nepochs{}_SFS".format(
            train_epoch).replace(
            ".", "") + ".parquet.gzip"
        bad_drift_df_1 = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"), engine="pyarrow")
        bad_drift_df_2 = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"), engine="pyarrow")
        good_drift_df = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_good.parquet.gzip"), engine="pyarrow")

        feature_data_path = config.experiment_path.replace("results/",
                                                           "results/feature_data/") + "_Nepochs{}_SFS".format(
            train_epoch).replace(".", "") + ".parquet.gzip"
        bad_feat_df_1 = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"),engine="pyarrow")
        bad_feat_df_2 = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"),engine="pyarrow")
        good_feat_df = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_good.parquet.gzip"),engine="pyarrow")

        bad_paths_df_1 = path_df.iloc[bad_drift_df_1.columns,:]
        print(bad_paths_df_1.columns)
        bad_paths_df_2 = path_df.iloc[bad_drift_df_2.columns,:]
        print(bad_paths_df_2.columns)
        good_paths_df = path_df.iloc[good_drift_df.columns,:]
        bad_hs_1 = hurst_estimation(bad_paths_df_1.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst)
        bad_hs_1.index = bad_paths_df_1.index
        bad_hs_2 = hurst_estimation(bad_paths_df_2.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst)
        bad_hs_2.index = bad_paths_df_2.index
        good_hs = hurst_estimation(good_paths_df.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst)
        good_hs.index = good_paths_df.index

        lsp = np.linspace(1, path_df.shape[1] + 1, path_df.shape[1])
        # Under-estimation
        for idx in bad_paths_df_1.index:
            path = bad_paths_df_1.loc[idx, :]
            bad_time = identify_jump_index(time_series=path, eps=1.1)
            if bad_time is not None: bad_path_times.append(bad_time)
            plt.plot(lsp, path, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
        plt.title("fBm Low Hurst Paths")
        plt.legend()
        plt.xlabel("Time")
        plt.show()
        # Over estimation
        for idx in bad_paths_df_2.index:
            path = bad_paths_df_2.loc[idx, :]
            bad_time = identify_jump_index(time_series=path, eps=1.1)
            if bad_time is not None: bad_path_times.append(bad_time)
            plt.plot(lsp, path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
        plt.title("fBm High Hurst Paths")
        plt.legend()
        plt.xlabel("Time")
        plt.show()
        # Now proceed to plot the drift score for the bad path indices
        # drift_error_df = pd.read_parquet(config.experiment_path.replace("results/", "results/drift_data/") + "_Nepochs{}_SFS".format(train_epoch).replace(".", "") + ".parquet.gzip", compression="gzip", index_col = [0,1], names=bad_path_idxs, engine="pyarrow")
        drift_error_df = pd.concat({i:pd.DataFrame(np.random.randn(config.max_diff_steps, len(bad_idxs))) for i in range(config.timeDim)})
        drift_error_df.columns = bad_idxs
        for bad_idxs in bad_path_idxs:
            for idx in range(len(bad_idxs)):
                drift_error_path = drift_error_df.loc[pd.IndexSlice[:, :] [idx]]
                plt.plot(lsp, drift_error_path, label=(bad_paths.index[idx], round(hs.iloc[bad_paths.index[idx], 0], 2)))
            plt.title("Drift Error for Bad Paths")
            plt.legend()
            plt.xlabel("Diffusion Time")
            plt.show()
            plt.close()
        # Now proceed to plot the feature time series for those paths
        # TODO: How to visualise 40 dimensional feature vector over whole real-time
        # feature_df = pd.read_parquet(config.experiment_path.replace("results/", "results/feature_data/") + "_Nepochs{}_SFS".format(train_epoch).replace(".", "") + ".parquet.gzip", compression="gzip", index_col = [0,1])
        feature_df = pd.concat({i:pd.DataFrame(np.random.randn(config.dataSize, 40)) for i in range(config.timeDim)})
        for bad_idxs in bad_path_idxs:
            fdf = feature_df.loc[pd.IndexSlice[:, bad_idxs], :]
            sns.boxplot(data=fdf)
            plt.title("fBm feature path for bad Hurst")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        good_path_idxs = hs.index.drop([idx for bad_idxs in bad_path_idxs for idx in bad_idxs])
        fdf = feature_df.loc[pd.IndexSlice[:, good_path_idxs], :]
        sns.boxplot(data=fdf)
        plt.title("fBm feature path for correct Hurst")
        plt.legend()
        plt.xlabel("Feature Dimension")
        plt.show()
if __name__ == "__main__":
    path_score_feature_analysis()
