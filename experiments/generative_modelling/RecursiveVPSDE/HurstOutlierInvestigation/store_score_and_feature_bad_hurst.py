import pandas as pd
from utils.plotting_functions import hurst_estimation


def store_score_and_feature_bad_hurst() -> None:
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    train_epoch = 1920

    path_df_path = config.experiment_path + "_Nepochs{}_SFS.parquet.gzip".format(train_epoch)
    path_df = pd.read_parquet(path_df_path, engine="pyarrow")
    print(path_df)
    hs = hurst_estimation(path_df.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch),
                          isfBm=config.isfBm, true_hurst=config.hurst)
    hs.index = path_df.index
    # Under-estimation
    lower = 0.55
    upper = 0.85
    bad_idxs_1 = hs.index[hs.lt(lower).any(axis=1)].to_list()  # Path IDs which satisfy condition
    bad_idxs_2 = hs.index[hs.gt(upper).any(axis=1)].to_list()  # Path IDs which satisfy condition
    good_idxs_1 = hs.index[hs.gt(lower).any(axis=1)].to_list()
    good_idxs_2 = hs.index[hs.lt(upper).any(axis=1)].to_list()
    good_idxs = (list(set(good_idxs_1) & set(good_idxs_2)))[:40]

    drift_data_path = config.experiment_path.replace("results/",
                                                     "results/drift_data/") + "_Nepochs{}_SFS".format(
        train_epoch).replace(
        ".", "") + ".parquet.gzip"
    drift_df = pd.read_parquet(drift_data_path, engine="pyarrow")
    bad_drift_df_1 = drift_df.loc[pd.IndexSlice[:,:],bad_idxs_1]
    bad_drift_df_1.to_parquet(drift_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"), compression="gzip")
    bad_drift_df_2 = drift_df.loc[pd.IndexSlice[:,:],bad_idxs_2]
    bad_drift_df_2.to_parquet(drift_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"), compression="gzip")
    good_drift_df = drift_df.loc[pd.IndexSlice[:,:],good_idxs]
    good_drift_df.to_parquet(drift_data_path.replace(".parquet.gzip", "_good.parquet.gzip"), compression="gzip")
    del drift_df

    feature_data_path = config.experiment_path.replace("results/", "results/feature_data/") + "_Nepochs{}_SFS".format(
        train_epoch).replace(".", "") + ".parquet.gzip"
    feature_df = pd.read_parquet(feature_data_path, engine="pyarrow")
    bad_feat_df_1 = feature_df.loc[pd.IndexSlice[:,bad_idxs_1],:]
    bad_feat_df_1.to_parquet(feature_data_path.replace(".parquet.gzip", "_bad1.parquet.gzip"), compression="gzip")
    bad_feat_df_2 = feature_df.loc[pd.IndexSlice[:,bad_idxs_2],:]
    bad_feat_df_2.to_parquet(feature_data_path.replace(".parquet.gzip", "_bad2.parquet.gzip"), compression="gzip")
    good_feat_df = feature_df.loc[pd.IndexSlice[:,good_idxs],:]
    good_feat_df.to_parquet(feature_data_path.replace(".parquet.gzip", "_good.parquet.gzip"), compression="gzip")
    del feature_df

if __name__ == "__main__":
    store_score_and_feature_bad_hurst()
