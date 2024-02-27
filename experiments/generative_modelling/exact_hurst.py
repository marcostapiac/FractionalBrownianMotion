import numpy as np
from utils.plotting_functions import hurst_estimation


def exact_hurst():
    hurst = 0.7
    exact = np.load("/Users/marcos/GitHubRepos/FractionalBrownianMotion/data/fBn_samples_H07_T256.npy")[:40000, :].cumsum(axis=1)
    hurst_estimation(exact, sample_type="Exact Samples",
                          isfBm=True, true_hurst=hurst)
if __name__ == "__main__":
    exact_hurst()