import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ml_collections import ConfigDict

from utils.plotting_functions import plot_histogram


def simple_estimator(paths: np.ndarray, Nsamples: int):
    # Filter 1 : (-1,1)
    U_a1 = np.mean(np.power(paths[:, 1:] - paths[:, :-1], 2), axis=1)[:, np.newaxis]
    assert (U_a1.shape == (Nsamples, 1))
    # Filter 2: (-1, 0, 1)
    U_a2 = np.mean(np.power(paths[:, 2:] - paths[:, :-2], 2), axis=1)[:, np.newaxis]
    assert (U_a1.shape == (Nsamples, 1))
    return U_a1, U_a2


def second_order_estimator(paths: np.ndarray, Nsamples: int):
    # Filter 1: (1,-2,1)
    U_a1 = np.mean(np.power(paths[:, 2:] - 2 * paths[:, 1:-1] + paths[:, :-2], 2), axis=1)[:, np.newaxis]
    assert (U_a1.shape == (Nsamples, 1))
    # Filter 2: (1, 0, -2, 0, 1)
    U_a2 = np.mean(np.power(paths[:, 4:] - 2 * paths[:, 2:-2] + paths[:, :-4], 2), axis=1)[:, np.newaxis]
    assert (U_a2.shape == (Nsamples, 1))
    return U_a1, U_a2


def estimate_hurst_from_filter(Ua1: np.ndarray, Ua2: np.ndarray):
    assert (Ua1.shape == Ua2.shape)
    hs = 0.5 * np.log2(Ua2 / Ua1)
    plot_histogram(hs, num_bins=150, xlabel="H", ylabel="density",
                   plottitle="Histogram of {} {} samples' estimated Hurst parameter")
    plt.show()
    return hs


def estimate_fSDEs(config: ConfigDict):
    fOU = pd.read_csv("/Users/marcos/GitHubRepos/FractionalBrownianMotion/experiments/results/rec_TSM_False_incs_True_unitIntv_fOU_VPSDE_model_H7000e-01_T256_Ndiff10000_Tdiff1000e+00_trainEps1e-04_BetaMax20000e+01_BetaMin10000e-04_DiffEmbSize64_ResLay10_ResChan8_DiffHiddenSize64_TrueHybrid_TrueWghts_LSTM_H20_Nlay1_fOU10_tl5_NEp480.csv.gzip", compression="gzip", index_col=[0,1]).to_numpy()
    # We want to construct first an estimator for H based on sample paths
    for _ in range(10):
        idx = np.random.randint(0, fOU.shape[0])
        plt.plot(np.linspace(0, 1, config.ts_length), fOU[idx,:])
    plt.show()
    plt.close()
    N, T = fOU.shape
    assert (config.hurst < 0.75)
    U_a1, U_a2 = simple_estimator(paths=fOU, Nsamples=N)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2)
    U_a1, U_a2 = second_order_estimator(paths=fOU, Nsamples=N)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2)
    fOU = np.load(config.data_path, allow_pickle=True)
    N, T = fOU.shape
    assert (config.hurst < 0.75)
    U_a1, U_a2 = simple_estimator(paths=fOU, Nsamples=N)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2)
    U_a1, U_a2 = second_order_estimator(paths=fOU, Nsamples=N)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2)


def check_hurst_condition_estimator(H: float):
    assert (0 < H < 1)
    l1 = 1. / (4. - 4 * H)
    l2 = (2 * H - 1) / (2. - 2 * H)
    assert (max(l1, l2) <= 1)


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    check_hurst_condition_estimator(H=config.hurst)
    estimate_fSDEs(config=config)
