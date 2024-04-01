import numpy as np

from configs import project_config

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    sig1 = np.load(config.feat_path.replace(".npy", "_1.npy"))
    sig2 = np.load(config.feat_path.replace(".npy", "_2.npy"))
    sig3 = np.load(config.feat_path.replace(".npy", "_3.npy"))
    sig4 = np.load(config.feat_path.replace(".npy", "_4_5.npy"))
    sig5 = np.load(config.feat_path.replace(".npy", "_6.npy"))
    sig6 = np.load(config.feat_path.replace(".npy", "_7_8.npy"))
    sig7 = np.load(config.feat_path.replace(".npy", "_9_10.npy"))
    sig8 = np.load(config.feat_path.replace(".npy", "_11.npy"))
    sig9 = np.load(config.feat_path.replace(".npy", "_12.npy"))
    sig10 = np.load(config.feat_path.replace(".npy", "_13_14.npy"))
    sigs = np.concatenate([sig1, sig2,sig3,sig4,sig5,sig6,sig7, sig8, sig9, sig10], axis=0)
    np.save(project_config.ROOT_DIR + "data/fBm_H{}_T{}_SigTrunc{}_SigDim{}.npy".format(str(config.hurst).replace(".", ""),
                                                                                                   config.ts_length,
                                                                                                   config.sig_trunc,
                                                                                                   config.sig_dim), sigs, allow_pickle=True)


