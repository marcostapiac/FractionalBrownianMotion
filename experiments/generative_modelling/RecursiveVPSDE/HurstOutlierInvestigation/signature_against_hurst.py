from pickle import UnpicklingError

import numpy as np
import torch
from matplotlib import pyplot as plt

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalSignatureTimeSeriesScoreMatching import \
    ConditionalSignatureTimeSeriesScoreMatching
from utils.math_functions import compute_sig_size

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    rng = np.random.default_rng()
    fbm_feats_path = project_config.ROOT_DIR+"/experiments/results/feature_data/true_fBm_features.npy"
    bm_feats_path = project_config.ROOT_DIR+"/experiments/results/feature_data/true_Bm_features.npy"
    try:
        fbm_feats = np.load(fbm_feats_path)
        bm_feats = np.load(bm_feats_path)
    except (FileNotFoundError, UnpicklingError) as e:
        data_shape = (5000, config.ts_length, 1)
        true_fBm = np.array([FractionalBrownianNoise(H=config.hurst, rng=rng).circulant_simulation(N_samples=config.ts_length).cumsum() for _ in range(data_shape[0])]).reshape((data_shape[0], data_shape[1]))[:,:,np.newaxis]
        true_Bm = np.array([FractionalBrownianNoise(H=0.5, rng=rng).circulant_simulation(N_samples=config.ts_length).cumsum() for _ in range(data_shape[0])]).reshape((data_shape[0], data_shape[1]))[:,:,np.newaxis]
        if config.has_cuda:
            # Sampling is sequential, so only single-machine, single-GPU/CPU
            device = 0
        else:
            device = torch.device("cpu")
        scoreModel = ConditionalSignatureTimeSeriesScoreMatching(*config.model_parameters)
        scoreModel.eval()
        scoreModel.to(device)
        with torch.no_grad():
            if isinstance(device, int):
                true_fBm_features = scoreModel.signet.forward(torch.Tensor(true_fBm).to(device), time_ax=torch.atleast_2d(
                    (torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device), basepoint=True)[:,:-1,:]
                true_Bm_features = scoreModel.signet.forward(torch.Tensor(true_Bm).to(device), time_ax=torch.atleast_2d(
                    (torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device), basepoint=True)[:, :-1, :]
            else:
                true_fBm_features = scoreModel.signet.forward(torch.Tensor(true_fBm).to(device),
                                                             time_ax=torch.atleast_2d((torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device),
                                                             basepoint=True)[:,:-1,:]
                true_Bm_features = scoreModel.signet.forward(torch.Tensor(true_Bm).to(device), time_ax=torch.atleast_2d(
                    (torch.arange(1, config.ts_length + 1) / config.ts_length)).T.to(device), basepoint=True)[:, :-1, :]
        np.save(project_config.ROOT_DIR+"/experiments/results/feature_data/true_fBm_features.npy", true_fBm_features.cpu().numpy())
        np.save(project_config.ROOT_DIR+"/experiments/results/feature_data/true_Bm_features.npy", true_Bm_features.cpu().numpy())
        print(true_fBm_features, true_Bm_features.shape)
        print("Done saving\n")
    else:
        avg_fbm_feats = np.mean(fbm_feats, axis=1)
        avg_bm_feats = np.mean(bm_feats, axis=1)
        feat_dim = compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc)-1
        dimspace = np.arange(1, feat_dim + 1, dtype=int)
        for t in range(1, config.ts_length):
            plt.plot(dimspace, avg_good_feat_df[t, :], label="Good Sim Path Sig Feat")
            # plt.plot(dimspace,avg_bad_feat_df[t,:], label="Bad Sim Path Sig Feat")
            # plt.plot(dimspace,avg_true_feat_df[t,:], label="Good True Path Sig Feat")
            plt.title("Sig Feat for history of TS time {}".format(t))
            plt.legend()
            plt.show()
            plt.close()
            break