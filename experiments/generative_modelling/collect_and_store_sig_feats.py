import numpy as np

from configs import project_config

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    sig = np.load(config.feat_path)
    print(sig)
    """sig1 = np.load(config.feat_path.replace(".npy", "_1.npy"))
    sig1b = np.load(config.feat_path.replace(".npy", "_1b.npy"))
    sig2 = np.load(config.feat_path.replace(".npy", "_2.npy"))
    sig3 = np.load(config.feat_path.replace(".npy", "_3.npy"))
    sig4 = np.load(config.feat_path.replace(".npy", "_4.npy"))
    sig5a = np.load(config.feat_path.replace(".npy", "_5a.npy"))
    sig5b = np.load(config.feat_path.replace(".npy", "_5b.npy"))
    sig6 = np.load(config.feat_path.replace(".npy", "_6.npy"))
    sig7a = np.load(config.feat_path.replace(".npy", "_7a.npy"))
    sigs = np.concatenate([sig, sig1,sig1b, sig2,sig3,sig4,sig5a,sig5b,sig6, sig7a], axis=0)
    np.save(project_config.ROOT_DIR + "data/fBm_H{}_T{}_SigTrunc{}_SigDim{}.npy".format(str(config.hurst).replace(".", ""),
                                                                                                   config.ts_length,
                                                                                                   config.sig_trunc,
                                                                                                   config.sig_dim), sigs, allow_pickle=True)"""


