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
        exact_feat_df = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_.parquet.gzip"),engine="pyarrow")
        bad_paths_df_1 = path_df.iloc[bad_drift_df_1.columns,:]
        bad_paths_df_2 = path_df.iloc[bad_drift_df_2.columns,:]
        good_paths_df = path_df.iloc[good_drift_df.columns,:]
        bad_hs_1 = hurst_estimation(bad_paths_df_1.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst, show=False)
        bad_hs_1.index = bad_paths_df_1.index
        bad_hs_2 = hurst_estimation(bad_paths_df_2.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst, show=False)
        bad_hs_2.index = bad_paths_df_2.index
        good_hs = hurst_estimation(good_paths_df.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                              isfBm=config.isfBm, true_hurst=config.hurst, show=False)
        good_hs.index = good_paths_df.index

        lsp = np.arange(1, path_df.shape[1] + 1)
        bad_path_times = {idx: None for idx in np.concatenate([bad_paths_df_1.index.values, bad_paths_df_2.index.values])}
        # Under-estimation
        for idx in bad_paths_df_1.index:
            path = bad_paths_df_1.loc[idx, :]
            bad_time = identify_jump_index(time_series=path, eps=1.1)
            if bad_time is not None: bad_path_times[idx] = bad_time
            plt.plot(lsp, path, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
        plt.title("fBm Low Hurst Paths")
        plt.legend()
        plt.xlabel("Time")
        plt.show()
        # Over estimation
        for idx in bad_paths_df_2.index:
            path = bad_paths_df_2.loc[idx, :]
            bad_time = identify_jump_index(time_series=path, eps=0.8)
            if bad_time is not None: bad_path_times[idx] = bad_time
            plt.plot(lsp, path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
        plt.title("fBm High Hurst Paths")
        plt.legend()
        plt.xlabel("Time")
        plt.show()
        # Keep only paths for which a time has been identified
        bad_path_times = {k: v for k, v in bad_path_times.items() if v is not None}
        bad_path_times = {392:151,7083:27, 17861:44, 24050:207,37797:170}
        # Now proceed to plot the drift score for the bad path indice
        diff_lsp = np.arange(1, config.max_diff_steps+1)
        feat_lsp = np.arange(1, 40+1)
        for idx in bad_paths_df_1.index:
            if bad_path_times[idx] != None:
                bad_drift_path_before = bad_drift_df_1.loc[pd.IndexSlice[bad_path_times[idx], :], [idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, bad_drift_path_before, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Under-estimated Hurst Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                bad_drift_path_at = bad_drift_df_1.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, bad_drift_path_at, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Under-estimated Hurst Paths At Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                bad_drift_path_at = bad_drift_df_1.loc[pd.IndexSlice[bad_path_times[idx] + 2, :], [idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, bad_drift_path_at, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Under-estimated Hurst Paths After Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                # Plot feature values at those three times
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx]-1, idx],:].values, label=("Before", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx], idx],:].values, label=("At", idx,round(bad_hs_1.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx]+1, idx],:].values, label=("After", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx]+15, idx],:].values, label=("Wrong", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Choose a random "good path" for comparison
                good_idx = np.random.choice(good_paths_df.index.values)
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx], :], [good_idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths At Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx] + 2, :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths After Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                # Plot feature values at those three times
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx]-1, good_idx],:].values, label=("Before", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx], good_idx],:].values, label=("At", good_idx,round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx]+1, good_idx],:].values, label=("After", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
        for idx in bad_paths_df_2.index:
            if bad_path_times[idx] != None:
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx], :], [idx]].droplevel(0).iloc[::-1].cumsum()/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths At Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx] + 2, :], [idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths After Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                # Plot feature values at those three times
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx]-1, idx],:].values, label=("Before", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx], idx],:].values, label=("At", idx,round(bad_hs_2.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx]+1, idx],:].values, label=("After", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx]+15, idx],:].values, label=("Wrong", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Choose a random "good path" for comparison
                good_idx = np.random.choice(good_paths_df.index.values)
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx] , :], [good_idx]].droplevel(0).iloc[::-1].cumsum().iloc[::-1]/np.arange(1, config.max_diff_steps+1 ,config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths At Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx] + 2, :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, good_drift_error, label=(good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("CumMean Drift Error for Good Paths After Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                # Plot feature values at those three times
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx] - 1, good_idx], :].values,
                         label=("Before", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx], good_idx], :].values,
                         label=("At", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx] + 1, good_idx], :].values,
                         label=("After", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Plot feature_values at those three times for "exact path" features
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx] - 1, good_idx], :].values,
                         label=("Before", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx], good_idx], :].values,
                         label=("At", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.plot(feat_lsp, good_feat_df.loc[pd.IndexSlice[bad_path_times[idx] + 1, good_idx], :].values,
                         label=("After", good_idx, round(good_hs.loc[good_idx][0], 2)))
                plt.title("Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
        # Now proceed to plot the feature time series for those paths
        # TODO: I want to know if the higher overall drift errors at jump times are due to an odd feature...
        # TODO: Therefore, for the time before and at the jump, I can plot the feature values
        """bfdf1 = bad_feat_df_1.groupby(level=1).apply(lambda x: x.loc[pd.IndexSlice[bad_path_times[
                                                                                                    x.index.unique(
                                                                                                        level=1)[
                                                                                                        0]] - 1:
                                                                                                bad_path_times[
                                                                                                    x.index.unique(
                                                                                                        level=1)[
                                                                                                        0]] + 1, :],
                                                               :].droplevel(1) if bad_path_times[x.index.unique(level=1)[
            0]] is not None else None)
        if not bfdf1.empty:
            sns.boxplot(data=bfdf1)
            plt.title("fBm Features for Under-estimated Hurst Near Jump Times")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        good_idxs = np.random.choice(good_paths_df.columns.values,size=bad_paths_df_1.shape[1])
        gfdf1 = good_feat_df.groupby(level=1).apply(lambda x: x.loc[pd.IndexSlice[bad_path_times[bad_paths_df_1.columns.to_list()] - 1:bad_path_times[bad_paths_df_1.columns.to_list()] + 1, good_idxs],:].droplevel(1))
        if not gfdf1.empty:
            sns.boxplot(data=gfdf1)
            plt.title("fBm feature path for Good Hurst Near Under-estimated Hurst Jump Times")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()

        bfdf2 = bad_feat_df_2.groupby(level=1).apply(lambda x: x.loc[pd.IndexSlice[bad_path_times[
                                                                                       x.index.unique(
                                                                                           level=1)[
                                                                                           0]] - 1:
                                                                                   bad_path_times[
                                                                                       x.index.unique(
                                                                                           level=1)[
                                                                                           0]] + 1, :],
                                                               :].droplevel(1) if bad_path_times[
                                                                                      x.index.unique(level=1)[
                                                                                          0]] is not None else None)
        if not bfdf2.empty:
            sns.boxplot(data=bfdf2)
            plt.title("fBm feature path for Over-estimated Hurst Near Jump Times")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        """

        # Now proceed to plot feature box plots over paths and times
        sns.boxplot(data=bad_feat_df_1)
        plt.title("fBm Features for Under-estimated Hurst")
        plt.legend()
        plt.xlabel("Feature Dimension")
        plt.show()
        sns.boxplot(data=bad_feat_df_2)
        plt.title("fBm feature path for Over-estimated Hurst")
        plt.legend()
        plt.xlabel("Feature Dimension")
        plt.show()
        sns.boxplot(data=good_feat_df)
        plt.title("fBm feature path for correct Hurst")
        plt.legend()
        plt.xlabel("Feature Dimension")
        plt.show()
if __name__ == "__main__":
    path_score_feature_analysis()
