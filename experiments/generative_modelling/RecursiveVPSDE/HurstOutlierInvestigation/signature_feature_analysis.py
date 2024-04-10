import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils.math_functions import compute_sig_size


def sig_path_score_feature_analysis() -> None:
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    # Now plot Hurst histogram for the generated samples
    for train_epoch in [960]:  # config.max_epochs:
        paths = pd.read_parquet("/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/rec_TSM_False_incs_True_unitIntv_fBm_VPSDE_model_H7000e-01_T256_Ndiff10000_Tdiff1000e+00_trainEps1e-04_BetaMax20000e+01_BetaMin10000e-04_DiffEmbSize64_ResLay10_ResChan8_DiffHiddenSize64_TrueHybrid_TrueWghts_Sig_Trunc5_Dim2_tl5_NEp960_SFS.parquet.gzip".format(train_epoch), engine="pyarrow")
        good_feat_df = pd.read_parquet("/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/feature_data/fBm_H07_T256_SigTrunc5_SigDim2_NEp{}_SFS_good.parquet.gzip".format(train_epoch), engine="pyarrow")
        bad_feat_df = pd.read_parquet("/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/feature_data/fBm_H07_T256_SigTrunc5_SigDim2_NEp{}_SFS_bad1.parquet.gzip".format(train_epoch), engine="pyarrow")
        true_feat_df = pd.read_parquet("/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/feature_data/fBm_H07_T256_SigTrunc5_SigDim2_True_NEp{}_SFS_good.parquet.gzip".format(train_epoch), engine="pyarrow")
        feat_dim = compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc)-1
        avg_true_feat_df = np.mean(true_feat_df.to_numpy().reshape((config.ts_length, true_feat_df.index.levshape[1],feat_dim)), axis=1)
        avg_good_feat_df = np.mean(good_feat_df.to_numpy().reshape((config.ts_length, good_feat_df.index.levshape[1],feat_dim)), axis=1)
        avg_bad_feat_df = np.mean(bad_feat_df.to_numpy().reshape((config.ts_length, bad_feat_df.index.levshape[1],feat_dim)), axis=1)
        assert(avg_true_feat_df.shape == (config.ts_length, feat_dim) and avg_good_feat_df.shape == avg_true_feat_df.shape and avg_bad_feat_df.shape == avg_true_feat_df.shape)
        # For each time, plot the average feature again
        dimspace = np.arange(1, feat_dim+1, dtype=int)
        for t in range(1,config.ts_length):
            plt.plot(dimspace,avg_good_feat_df[t,:], label="Good Sim Path Sig Feat")
            plt.plot(dimspace,avg_bad_feat_df[t,:], label="Bad Sim Path Sig Feat")
            plt.plot(dimspace,avg_true_feat_df[t,:], label="Good True Path Sig Feat")
            plt.title("Sig Feat for history of TS time {}".format(t))
            plt.legend()
            #plt.show()
            plt.close()

        bad_idxs = bad_feat_df.index.unique(level=1)
        for bad_idx in bad_idxs:
            plt.plot(np.arange(1, config.ts_length+1), paths.loc[bad_idx,:])
            print(bad_idx)
            break
        plt.show()
        plt.close()
        good_idxs = good_feat_df.index.unique(level=1)
        for good_idx in good_idxs:
            plt.plot(np.arange(1, config.ts_length+1), paths.loc[good_idx,:])
            print(good_idx)
            break
        plt.show()
        plt.close()


if __name__=="__main__":
    sig_path_score_feature_analysis()