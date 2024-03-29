import numpy as np
import roughpy as rhpy

from configs import project_config
from utils.math_functions import time_aug, assert_chen_identity, compute_signature, invisibility_reset, \
    ts_signature_pipeline, compute_sig_size

if __name__ == "__main__":
    # Example usage
    trunc = 3
    fBm_data = np.atleast_3d(np.load(project_config.ROOT_DIR + "/data/fBn_samples_H07_T8.npy")).cumsum(axis=1)[:2,
               :, :]
    N, T, d = fBm_data.shape
    times = np.atleast_2d((np.arange(1, T + 1) / T)).T
    feats = ts_signature_pipeline(data_batch=fBm_data,trunc=trunc, times=times)
    assert(feats.shape == (N, compute_sig_size(dim=d+1, trunc=trunc)))
    print(feats)
    # Now attempt on a rolling basis across time
    full_feats = np.zeros(shape=(N, T, compute_sig_size(dim=d+1, trunc=trunc)))
    for t in range(T):
        if t == 0:
            full_feats[:,t,:] = ts_signature_pipeline(data_batch=np.hstack([np.zeros(shape=(N, 1, d)),fBm_data[:, [t],:]]), trunc=trunc, times=times)
        else:
            full_feats[:,t,:] = ts_signature_pipeline(data_batch=fBm_data[:, :t,:], trunc=trunc, times=times)

    assert(np.all(np.abs(full_feats[:, -1, :] - feats))<1e-10)