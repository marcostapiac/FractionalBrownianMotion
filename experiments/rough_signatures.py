import numpy as np
import roughpy as rhpy

from configs import project_config
from utils.math_functions import time_aug, assert_chen_identity, compute_signature, invisibility_reset

if __name__ == "__main__":
    # Example usage
    fBm_data = np.atleast_3d(np.load(project_config.ROOT_DIR + "/data/fBn_samples_H07_T8.npy")).cumsum(axis=1)[:2,
               :, :]
    N, T, d = fBm_data.shape
    # Before working with RoughPy, transform our data into a stream of increments using Lyons and McLeod (2.8)
    #   1. Augmentation step first: the easiest is to innclude a time channel in our data
    timeaug = time_aug(fBm_data, np.atleast_2d(np.arange(1, T + 1) / T).T)
    #   2. We transform (xi0, ..., xiT) into (xi0, 0, xi1-xi0, xi2-xi1, xi3-xi2, ..., xiT-xiT-1, 0, -xiT)
    transformed = invisibility_reset(timeaug, ts_dim=d)
    dims = transformed.shape[-1]
    trunc = 3
    # Verify Chen's identity
    assert_chen_identity(sample=transformed[1, :, :], trunc=trunc, dim=dims, coefftype=rhpy.DPReal)
    feats = np.atleast_2d([compute_signature(sample=transformed[i, :, :], trunc=trunc, interval=rhpy.RealInterval(0, 1),
                                             dim=dims, coefftype=rhpy.DPReal) for i in range(N)])
    # Normalise features for machine learning
    print(np.mean(feats, axis=0))
    # feats = StandardScaler().fit_transform(feats)
    print(np.mean(feats, axis=0))
    print(feats, feats.shape)
