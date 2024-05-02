import time
from pickle import UnpicklingError

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_sig_size

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    rng = np.random.default_rng()
    feat_dim = compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1
    fbm_feats_path = project_config.ROOT_DIR + "/experiments/results/feature_data/true_fBm_features.npy"
    bm_feats_path = project_config.ROOT_DIR + "/experiments/results/feature_data/true_Bm_features.npy"
    train_epoch = 960
    try:
        fbm_feats = np.load(fbm_feats_path, allow_pickle=True)
        bm_feats = np.load(bm_feats_path, allow_pickle=True)
        sim_feat_df = pd.read_parquet(
            "/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/feature_data/fBm_H07_T256_SigTrunc5_SigDim2_True_NEp{}_SFS_bad1.parquet.gzip".format(
                train_epoch), engine="pyarrow")
    except (FileNotFoundError, UnpicklingError) as e:
        data_shape = (5000, config.ts_length, 1)
        true_fBm = np.array(
            [FractionalBrownianNoise(H=config.hurst, rng=rng).circulant_simulation(N_samples=config.ts_length).cumsum()
             for _ in range(data_shape[0])]).reshape((data_shape[0], data_shape[1]))[:, :, np.newaxis]
        true_Bm = np.array(
            [FractionalBrownianNoise(H=0.5, rng=rng).circulant_simulation(N_samples=config.ts_length).cumsum() for _ in
             range(data_shape[0])]).reshape((data_shape[0], data_shape[1]))[:, :, np.newaxis]
        if config.has_cuda:
            # Sampling is sequential, so only single-machine, single-GPU/CPU
            device = 0
        else:
            device = torch.device("cpu")
        scoreModel = ConditionalSignatureTSScoreMatching(*config.model_parameters)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
        scoreModel.eval()
        scoreModel.to(device)
        with torch.no_grad():
            if isinstance(device, int):
                true_fBm_features = scoreModel.signet.forward(torch.Tensor(true_fBm).to(device),
                                                              time_ax=torch.atleast_2d(
                                                                  (torch.arange(1,
                                                                                config.ts_length + 1) / config.ts_length)).T.to(
                                                                  device), basepoint=True)[:, :-1, :]
                true_Bm_features = scoreModel.signet.forward(torch.Tensor(true_Bm).to(device), time_ax=torch.atleast_2d(
                    (torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device), basepoint=True)[:, :-1, :]
            else:
                true_fBm_features = scoreModel.signet.forward(torch.Tensor(true_fBm).to(device),
                                                              time_ax=torch.atleast_2d((torch.arange(1,
                                                                                                     config.ts_length + 1) / config.ts_length)).T.to(
                                                                  device),
                                                              basepoint=True)[:, :-1, :]
                true_Bm_features = scoreModel.signet.forward(torch.Tensor(true_Bm).to(device), time_ax=torch.atleast_2d(
                    (torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device), basepoint=True)[:, :-1, :]
        np.save(project_config.ROOT_DIR + "/experiments/results/feature_data/true_fBm_features.npy",
                true_fBm_features.cpu().numpy())
        np.save(project_config.ROOT_DIR + "/experiments/results/feature_data/true_Bm_features.npy",
                true_Bm_features.cpu().numpy())
        print(true_fBm_features, true_Bm_features.shape)
        print("Done saving\n")
    else:
        avg_fbm_feats = np.mean(fbm_feats, axis=0)
        avg_bm_feats = np.mean(bm_feats, axis=0)
        avg_true_feat_df = np.mean(
            sim_feat_df.to_numpy().reshape((config.ts_length, sim_feat_df.index.levshape[1], feat_dim)), axis=1)

        assert (avg_bm_feats.shape == avg_fbm_feats.shape == avg_true_feat_df.shape and avg_bm_feats.shape == (
            config.ts_length, feat_dim))
        dimspace = np.arange(1, feat_dim + 1, dtype=int)
        for t in range(1, config.ts_length):
            plt.plot(dimspace, avg_fbm_feats[t, :], label="fBm Sig Feat")
            plt.plot(dimspace, avg_bm_feats[t, :], label="Bm Sig Feat")
            plt.plot(dimspace, avg_true_feat_df[t, :], label="Sim fBm Sig Feat")

            plt.title("Sig Feat for history of TS time {}".format(t))
            plt.legend()
            plt.show()
            plt.close()
            time.sleep(1)
