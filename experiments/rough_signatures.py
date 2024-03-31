import numpy as np
import roughpy as rhpy
import torch

from configs import project_config
from utils.math_functions import time_aug, assert_chen_identity, compute_signature, invisibility_reset, \
    ts_signature_pipeline, compute_sig_size

if __name__ == "__main__":
    # Example usage
    trunc = 3
    fBm_data = np.atleast_3d(np.load(project_config.ROOT_DIR + "/data/fBn_samples_H07_T8.npy")).cumsum(axis=1)[:1,
               :, :]
    fBm_data = torch.Tensor(np.atleast_3d(fBm_data))
    N, T, d = fBm_data.shape
    times = torch.atleast_2d((torch.arange(1, T + 1) / T)).T.to(fBm_data.device)
    feats = ts_signature_pipeline(data_batch=fBm_data,trunc=trunc, times=times)
    assert(feats.shape == (N, compute_sig_size(dim=d+1, trunc=trunc)))

