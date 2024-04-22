import time

import pandas as pd
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from experiments.generative_modelling.estimate_fSDEs import estimate_fSDE_from_true, second_order_estimator, \
    estimate_hurst_from_filter


def estimate_SDEs(config: ConfigDict, train_epoch: int) -> None:
    incs = pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip",
                       index_col=[0, 1]).to_numpy()
    means = pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUm"),
                        compression="gzip", index_col=[0, 1]).to_numpy()
    vars = pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUv"), compression="gzip",
                       index_col=[0, 1]).to_numpy()
    paths = incs.cumsum(axis=1)
    # Compare marginal distributions
    time_space = np.linspace(0, 1. + (1. / config.ts_length), num=config.ts_length)[1:]
    sidx = 254
    for tidx in range(sidx, config.ts_length):
        t = time_space[tidx]
        expmeanrev = np.exp(-config.mean_rev * t)
        exp_mean = config.mean * (1. - expmeanrev)
        exp_mean += config.initState * expmeanrev
        exp_var = np.power(config.diffusion, 2)
        exp_var /= (2 * config.mean_rev)
        exp_var *= 1. - np.power(expmeanrev, 2)
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=paths.shape[0])
        pathst = paths[:, tidx]
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title("Marginal Distributions at time {}".format(t))
        plt.legend()
        plt.show()
        plt.close()
    # Estimate Hurst indices from paths
    U_a1, U_a2 = second_order_estimator(paths=paths, Nsamples=paths.shape[0])
    path_hs = estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=train_epoch).flatten()
    for _ in range(1000):
        idx = np.random.randint(0, paths.shape[0])
        plt.plot(np.linspace(0, 1, config.ts_length), paths[idx, :])
    plt.show()
    plt.close()
    # Estimate Hurst indices from true
    true_hs = estimate_fSDE_from_true(config=config).flatten()
    # Remove any path with hurst index outside of range
    # delidx = np.array([(min(true_hs) <= path_hs[i] <= max(true_hs)) for i in range(path_hs.shape[0])])
    # paths = paths.to_numpy()[delidx,:]
    # means = means.to_numpy()[delidx,:]
    # vars = vars.to_numpy()[delidx,:]
    # Now use EM approximation to compute the instantaneous parameters at every point in time and space
    vars *= config.ts_length
    means *= config.ts_length
    ys = means.flatten().reshape((means.shape[0] * means.shape[1], 1))
    designmat = incs.flatten().reshape((incs.shape[0] * incs.shape[1], 1))
    meanrev = -np.linalg.solve(designmat.T @ designmat, designmat.T @ ys)
    print(meanrev)
    print(np.mean(vars), np.std(vars))
    # Now plot histograms across time and space of estimates
    for t in range(254, config.ts_length):
        meant = means[:, t]
        varst = vars[:, t]
        pathst = paths[:, t]
        plt.plot(pathst, meant)
        plt.title(f"Drift Function at time {t + 1} against state value")
        plt.show()
        plt.close()
        plt.hist(meant, bins=150, density=True)
        plt.title(f"Instantaneous Drift Estimates at time {t + 1}")
        plt.show()
        plt.close()
        plt.plot(pathst, varst)
        plt.title(f"Diffusion Function at time {t + 1} against state value")
        plt.show()
        plt.close()
        plt.hist(varst, bins=150, density=True)
        plt.title(f"Instantaneous Volatility Estimate at time {t + 1}")
        plt.show()
        plt.close()
        time.sleep(0.5)
    print("HI")


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    for train_epoch in config.max_epochs:
        print(train_epoch)
        train_epoch = 2920
        estimate_SDEs(config=config, train_epoch=train_epoch)
