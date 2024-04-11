import numpy as np

from utils.plotting_functions import hurst_estimation


def exact_hurst():
    hurst = 0.7
    exact = np.load("/Users/marcos/GitHubRepos/FractionalBrownianMotion/data/fBn_samples_H07_T256.npy")[:40000,
            :].cumsum(axis=1)
    hurst_estimation(exact, sample_type="Exact Samples",
                     isfBm=True, true_hurst=hurst)


if __name__ == "__main__":
    # exact_hurst()
    H = 0.75
    l1 = (2 * H - 1) / (2 - 2 * H)
    l2 = 1. / (4. - 4 * H)
    alpha = max(l1, max(l2, 1))
    print(alpha)
