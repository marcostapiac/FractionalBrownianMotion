import numpy as np
import roughpy as rhpy
import torch

from configs import project_config
from src.generative_modelling.data_processing import compute_current_sig_feature
from utils.data_processing import init_experiment, cleanup_experiment
from utils.math_functions import time_aug, assert_chen_identity, compute_signature, invisibility_reset, \
    ts_signature_pipeline, compute_sig_size

if __name__ == "__main__":
    # Example usage
    trunc = 3
    fBm_data = np.atleast_3d(np.load(project_config.ROOT_DIR + "/data/fBn_samples_H07_T8.npy")).cumsum(axis=1)[:2,
               :, :]
    fBm_data = torch.Tensor(np.atleast_3d(fBm_data))
    N, T, d = fBm_data.shape
    times = torch.atleast_2d((torch.arange(1, T + 1) / T)).T.to(fBm_data.device)
    feats = ts_signature_pipeline(data_batch=fBm_data,trunc=trunc, times=times)
    assert(feats.shape == (N, compute_sig_size(dim=d+1, trunc=trunc)))
    # Test concatenation property
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    fBm_data = fBm_data.to(device)
    init_experiment(config=config)
    cleanup_experiment()
    paths = []
    real_times = torch.atleast_2d((torch.arange(1, T + 1) / T)).T.to(device)
    for t in range(T):
        print("Sampling at real time {}\n".format(t + 1))
        if t == 0:
            output = torch.zeros(
                size=(N, 1, compute_sig_size(dim=d+1, trunc=trunc))).to(device)
            output[:, 0, 0] = 1.
        else:
            output = compute_current_sig_feature(ts_time=t, past_feat=output, latest_increment=paths[-2:],
                                                 config=config, real_times=real_times).to(device)

        samples = fBm_data[:,[t],:]
        assert (samples.shape == (N, 1, d))
        paths.append(samples)

