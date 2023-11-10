import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_fBn_cov, compute_fBm_cov

if __name__ == "__main__":
    # Data parameters
    hurst = 0.7
    timedim = 256
    rng = np.random.default_rng()
    time_space = np.linspace(0., 1., num=timedim + 1)[1:]
    fbn = FractionalBrownianNoise(H=hurst, rng=rng)

    fBn_cov = compute_fBn_cov(fbn, T=timedim, isUnitInterval=True)
    fBn_r_mat = np.diag([(timedim ** hurst) for ti in time_space]).reshape(timedim, timedim)
    fBn_corrmat = fBn_r_mat @ fBn_cov @ fBn_r_mat

    print("fBn Variance matrix:\n", np.linalg.inv(fBn_r_mat) @ np.linalg.inv(fBn_r_mat))
    print("fBn Covariance matrix:\n", fBn_cov)
    print("fBn Correlation matrix:\n", fBn_corrmat)

    print("\n\n")

    fBm_cov = (compute_fBm_cov(fbn, T=timedim, isUnitInterval=True))
    fBm_r_mat = np.diag([1. / (ti ** hurst) for ti in time_space]).reshape(timedim, timedim)
    fBm_corrmat = fBm_r_mat @ fBm_cov @ fBm_r_mat
    print("fBm Variance matrix:\n", np.linalg.inv(fBm_r_mat) @ np.linalg.inv(fBm_r_mat))
    print("fBm Covariance matrix:\n", fBm_cov)
    print("fBm Correlation matrix:\n", fBm_corrmat)
