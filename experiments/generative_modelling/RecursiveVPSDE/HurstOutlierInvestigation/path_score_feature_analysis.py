import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from ml_collections import ConfigDict

from experiments.generative_modelling.estimate_fSDEs import estimate_fSDEs
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


def feature_distribution(config: ConfigDict, feature_df: pd.DataFrame) -> None:
    assert (feature_df.index.levshape[0] == config.ts_length and feature_df.shape[1] == config.lstm_hiddendim)
    fbm_data = np.load(config.data_path, allow_pickle=True).cumsum(axis=1)[:feature_df.index.levshape[1]]
    # For every time "t", there exists a 20 dimensional feature.
    # For each (t, d), I want to plot the histogram of the feature values and overlay it against the histogram of
    xaxis = np.arange(0, config.lstm_hiddendim + 1)
    for t in range(config.ts_length):
        curr_feats = feature_df.loc[pd.IndexSlice[t, :], :].droplevel(0)
        avg_curr_feats = np.squeeze(curr_feats.to_numpy().mean(axis=0))
        std_curr_feats = np.squeeze(curr_feats.to_numpy().std(axis=0))
        curr_fbm = fbm_data[[t], :]
        avg_curr_fbm = np.squeeze(curr_fbm.mean(axis=0))
        std_curr_fbm = np.squeeze(curr_fbm.std(axis=0))
        assert (avg_curr_feats.shape == (config.lstm_hiddendim,))
        plt.plot(xaxis, avg_curr_feats, label="Average Learnt Feature")
        plt.fill_between(xaxis, avg_curr_feats - std_curr_feats, avg_curr_feats + std_curr_feats, color='C0', alpha=0.5)
        plt.plot(xaxis, avg_curr_fbm, label="Average Current FBM")
        plt.fill_between(xaxis, avg_curr_fbm - std_curr_fbm, avg_curr_fbm + std_curr_fbm, color='C1', alpha=0.5)
        plt.xlabel("Feature Dimension")
        plt.ylabel("Feature Value")
        plt.legend()
        plt.show()
        plt.close()


def path_score_feature_analysis() -> None:
    from configs.RecursiveVPSDE.recursive_LSTM_fOU_T256_H05_tl_5data import get_config
    config = get_config()
    # Now plot Hurst histogram for the generated samples
    for train_epoch in [960]:
        path_df_path = config.experiment_path + "_NEp{}_SFS.parquet.gzip".format(train_epoch)
        path_df = pd.read_parquet(path_df_path, engine="pyarrow")
        path_df = path_df
        assert (path_df.shape == (5000, 256))
        estimate_fSDEs(config=config, path=path_df_path, train_epoch=train_epoch)
        config.experiment_path = config.experiment_path.replace("LSTM_H40_Nlay2_", "")

        drift_data_path = (config.experiment_path.replace("results/",
                                                          "results/drift_data/") + "_NEp{}_SFS".format(
            train_epoch).replace(
            ".", "") + ".parquet.gzip").replace("rec_TSM_False_incs_True_unitIntv_", "")
        try:
            bad_drift_df_1 = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"),
                                             engine="pyarrow")
        except FileNotFoundError:
            bad_drift_df_1 = pd.DataFrame()
        try:
            bad_drift_df_2 = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"),
                                             engine="pyarrow")
        except FileNotFoundError:
            bad_drift_df_2 = pd.DataFrame()
        try:
            good_drift_df = pd.read_parquet(drift_data_path.replace(".parquet.gzip", "_good.parquet.gzip"),
                                            engine="pyarrow")
        except FileNotFoundError:
            good_drift_df = pd.DataFrame()

        feature_data_path = (config.experiment_path.replace("results/",
                                                            "results/feature_data/") + "_NEp{}_SFS".format(
            train_epoch).replace(".", "") + ".parquet.gzip").replace("rec_TSM_False_incs_True_unitIntv_", "")
        try:
            bad_feat_df_1 = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"),
                                            engine="pyarrow")
        except FileNotFoundError:
            bad_feat_df_1 = pd.DataFrame()
        try:
            bad_feat_df_2 = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"),
                                            engine="pyarrow")
        except FileNotFoundError:
            bad_feat_df_2 = pd.DataFrame()
        try:
            good_feat_df = pd.read_parquet(feature_data_path.replace(".parquet.gzip", "_good.parquet.gzip"),
                                           engine="pyarrow")
        except FileNotFoundError:
            good_feat_df = pd.DataFrame()
        try:
            exact_feat_df_all = pd.read_parquet(
                feature_data_path.replace("_SFS.parquet.gzip", "_Exact.parquet.gzip"), engine="pyarrow")
            exact_feat_df = exact_feat_df_all.groupby(level=0).apply(lambda x: x.droplevel(0).mean(axis=0))
        except FileNotFoundError:
            exact_feat_df = pd.DataFrame()
        lsp = np.arange(1, path_df.shape[1] + 1)
        bad_path_times = {idx: None for idx in
                          np.concatenate([bad_drift_df_1.columns.values, bad_drift_df_2.columns.values])}

        if not bad_drift_df_1.empty:
            bad_paths_df_1 = path_df.iloc[bad_drift_df_1.columns, :]
            bad_hs_1 = hurst_estimation(bad_paths_df_1.to_numpy(),
                                        sample_type="Under estimated Paths at Train Epoch {}".format(train_epoch),
                                        isfBm=config.isfBm, true_hurst=config.hurst, show=False)
            bad_hs_1.index = bad_paths_df_1.index
            # Under-estimation
            for idx in bad_paths_df_1.index:
                path = bad_paths_df_1.loc[idx, :]
                bad_time = identify_jump_index(time_series=path, eps=1.1)
                if bad_time is not None: bad_path_times[idx] = bad_time
                plt.plot(lsp, path, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
            plt.title("LSTM_fBm Low Hurst Paths")
            plt.legend()
            plt.xlabel("Time")
            plt.show()
        if not bad_drift_df_2.empty:
            bad_paths_df_2 = path_df.iloc[bad_drift_df_2.columns, :]
            bad_hs_2 = hurst_estimation(bad_paths_df_2.to_numpy(),
                                        sample_type="Over estimated Paths at Train Epoch {}".format(train_epoch),
                                        isfBm=config.isfBm, true_hurst=config.hurst, show=False)
            bad_hs_2.index = bad_paths_df_2.index
            # Over estimation
            for idx in bad_paths_df_2.index:
                path = bad_paths_df_2.loc[idx, :]
                bad_time = identify_jump_index(time_series=path, eps=0.8)
                if bad_time is not None: bad_path_times[idx] = bad_time
                plt.plot(lsp, path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
            plt.title("LSTM_fBm High Hurst Paths")
            plt.legend()
            plt.xlabel("Time")
            plt.show()
        if not good_drift_df.empty:
            good_paths_df = path_df.iloc[good_drift_df.columns, :]
            good_hs = hurst_estimation(good_paths_df.to_numpy(),
                                       sample_type="Good Paths at Train Epoch {}".format(train_epoch),
                                       isfBm=config.isfBm, true_hurst=config.hurst, show=False)
            good_hs.index = good_paths_df.index
            # Over estimation
            for idx in good_paths_df.index:
                path = good_paths_df.loc[idx, :]
                plt.plot(lsp, path, label=(idx, round(good_hs.loc[idx][0], 2)))
            plt.title("LSTM_fBm Good Hurst Paths")
            plt.legend()
            plt.xlabel("Time")
            plt.show()

        # Keep only paths for which a time has been identified
        bad_path_times = {8418: 253, 27805: 253, 27807: 253}
        # Now proceed to plot the drift score for the bad path indice
        diff_lsp = np.arange(1, config.max_diff_steps + 1)
        feat_lsp = np.arange(1, config.feat_hiddendim + 1)
        for idx in bad_drift_df_1.columns:
            if bad_path_times[idx] != None:
                bad_drift_path_before = bad_drift_df_1.loc[pd.IndexSlice[bad_path_times[idx], :], [idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, bad_drift_path_before, label=(idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Under-estimated Hurst Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                bad_drift_path_at = bad_drift_df_1.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
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
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx] - 1, idx], :].values,
                         label=("Before", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx], idx], :].values,
                         label=("At", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx] + 1, idx], :].values,
                         label=("After", idx, round(bad_hs_1.loc[idx][0], 2)))
                # plt.plot(feat_lsp, bad_feat_df_1.loc[pd.IndexSlice[bad_path_times[idx]+15, idx],:].values, label=("Wrong", idx, round(bad_hs_1.loc[idx][0], 2)))
                plt.title("Under-estimated Path Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Choose a random "good path" for comparison
                good_idx = np.random.choice(good_paths_df.index.values)
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx], :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
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
                plt.title("Good Path Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Plot "exact" feature values at those three times
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx] - 1, :].values,
                         label="Before")
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx], :].values,
                         label="At")
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx] + 1, :].values,
                         label="After")
                plt.title("Exact Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()

        for idx in bad_drift_df_2.columns:
            if bad_path_times[idx] != None:
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx], :], [idx]].droplevel(0).iloc[
                                   ::-1].cumsum() / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths Before Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx] + 1, :], [idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths At Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                drift_error_path = bad_drift_df_2.loc[pd.IndexSlice[bad_path_times[idx] + 2, :], [idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
                plt.plot(diff_lsp, drift_error_path, label=(idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("CumMean Drift Error for Over-estimated Hurst Paths After Jump Time")
                plt.legend()
                plt.xlabel("Diffusion Time")
                plt.show()
                plt.close()
                # Plot feature values at those three times
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx] - 1, idx], :].values,
                         label=("Before", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx], idx], :].values,
                         label=("At", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx] + 1, idx], :].values,
                         label=("After", idx, round(bad_hs_2.loc[idx][0], 2)))
                # plt.plot(feat_lsp, bad_feat_df_2.loc[pd.IndexSlice[bad_path_times[idx]+15, idx],:].values, label=("Wrong", idx, round(bad_hs_2.loc[idx][0], 2)))
                plt.title("Over-estimated Path Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Choose a random "good path" for comparison
                good_idx = np.random.choice(good_paths_df.index.values)
                good_drift_error = good_drift_df.loc[pd.IndexSlice[bad_path_times[idx], :], [good_idx]].droplevel(
                    0).iloc[::-1].cumsum().iloc[::-1] / np.arange(1, config.max_diff_steps + 1, config.max_diff_steps)
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
                plt.title("Good Path Feature Values Near Jump Times")
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
                plt.title("Good Path Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()
                # Plot "exact" feature values at those three times
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx] - 1, :].values,
                         label="Before")
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx], :].values,
                         label="At")
                plt.plot(feat_lsp, exact_feat_df.loc[bad_path_times[idx] + 1, :].values,
                         label="After")
                plt.title("Exact Feature Values Near Jump Times")
                plt.legend()
                plt.xlabel("Feature Dimension")
                plt.show()
                plt.close()

        # Now proceed to plot feature box plots over paths and times
        try:
            sns.boxplot(data=bad_feat_df_1)
            plt.title("LSTM_fBm Features for Under-estimated Hurst")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        except Exception:
            pass
        try:
            sns.boxplot(data=bad_feat_df_2)
            plt.title("LSTM_fBm feature path for Over-estimated Hurst")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        except Exception:
            pass
        try:
            sns.boxplot(data=good_feat_df)
            plt.title("LSTM_fBm feature path for correct Hurst")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        except Exception:
            pass
        try:
            sns.boxplot(data=exact_feat_df_all)
            plt.title("LSTM_fBm feature path for exact Hurst")
            plt.legend()
            plt.xlabel("Feature Dimension")
            plt.show()
        except Exception:
            pass
        avg_feat = good_feat_df.to_numpy().reshape(
            (good_feat_df.index.levshape[0], good_feat_df.index.levshape[1], good_feat_df.shape[1]))
        avg_feat = np.mean(avg_feat, axis=1)
        for i in range(avg_feat.shape[1]):
            plt.plot(np.arange(1, config.ts_length + 1), avg_feat[:, i], label="Dim {}".format(i + 1))
            plt.legend()
            plt.show()
            plt.close()


if __name__ == "__main__":
    path_score_feature_analysis()
