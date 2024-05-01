import time

import pandas as pd
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiments.generative_modelling.estimate_fSDEs import estimate_fSDE_from_true, second_order_estimator, \
    estimate_hurst_from_filter


def estimate_SDEs(config: ConfigDict, train_epoch: int) -> None:
    incs = pd.read_csv(config.experiment_path + "_rdNEp{}.csv.gzip".format(train_epoch), compression="gzip",
                       index_col=[0, 1]).to_numpy()
    paths = incs.cumsum(axis=1)
    for _ in range(paths.shape[0]):
        plt.plot(np.linspace(0, 1, config.ts_length), paths[_, :])
    plt.show()
    plt.close()
    # Estimate Hurst indices from paths
    U_a1, U_a2 = second_order_estimator(paths=paths, Nsamples=paths.shape[0])
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=train_epoch).flatten()
    # Estimate Hurst indices from true
    estimate_fSDE_from_true(config=config).flatten()
    # Construct estimator for mean reversion
    augdata = np.concatenate([np.zeros(shape=(paths.shape[0], 1)), paths], axis=1)
    denom = np.sum(np.power(augdata[:, :-1], 2), axis=1) * (1. / config.ts_length)
    diffs = np.diff(augdata, n=1, axis=1)
    num = np.sum(augdata[:, :-1] * diffs, axis=1)
    estimators = -num / denom
    plt.hist(estimators, bins=150, density=True)
    plt.vlines(x=config.mean_rev, ymin=0, ymax=0.25, label="", color='b')
    plt.title("Estimator")
    plt.legend()
    plt.show()
    plt.close()
    if config.param_time == config.max_diff_steps - 1:
        PT = 0
    elif config.param_time == 4600:
        PT = 2
    else:
        PT = 1
    means = pd.read_csv(
        (config.experiment_path + "_rdNEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU", "fOUm").replace("fOUm00", "fm00"),
        compression="gzip", index_col=[0, 1]).to_numpy()
    means *= (config.ts_length ** (2 * config.hurst))
    M = means.shape[0]
    N = means.shape[1] - 1
    mean_revs = []
    mmeans = means[:, 1:]
    for pathidx in tqdm(range(M)):
        ys = mmeans[pathidx, :].flatten().reshape((N, 1))
        designmat = paths[pathidx, :-1].flatten().reshape((N, 1))
        meanrev = -np.linalg.solve(designmat.T @ designmat, designmat.T @ ys)
        mean_revs.append(float(meanrev))
    plt.hist(mean_revs, bins=150, density=True)
    plt.title(f"Mean Reversion Linear Regression Estimates for epoch {train_epoch}")
    plt.show()
    plt.close()

    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
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
        pathst = paths[:, tidx]  # Paths[:, 0] corresponds to X_{t_{1}} NOT X_{t_{0}}
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title(f"Marginal Distributions at time {t} for epoch {train_epoch}")
        plt.legend()
        plt.show()
        plt.close()

    # Plot drift observations for each path
    for _ in range(10):
        idx = np.random.randint(low=0, high=paths.shape[0])
        path = paths[idx, :-1]
        meanp = mmeans[idx, :]
        plt.scatter(path, meanp)
        plt.plot(path, -config.mean_rev * path, color="blue")
        plt.title("Drift against Path")
        plt.show()
        plt.close()
        time.sleep(1)

    # Plot path and drift in same plot observations for each path
    for _ in range(10):
        idx = np.random.randint(low=0, high=paths.shape[0])
        mean = means[idx, :]
        path = paths[idx,:]
        plt.plot(time_space, mean, label="Drift")
        plt.plot(time_space, path, color="blue", label="Path")
        plt.title("Path/Drift against Time")
        plt.legend()
        plt.show()
        plt.close()
        time.sleep(1)


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    train_epoch = 2920
    for param_time in [900,4600, 9999]:
        try:
            pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip",
                        index_col=[0, 1]).to_numpy()
            train_epoch = 2920
            config.param_time = param_time
            if config.param_time == config.max_diff_steps - 1:
                PT = 0
            elif config.param_time == 4600:
                PT = 2
            else:
                PT = 1
            estimate_SDEs(config=config, train_epoch=train_epoch)
        except FileNotFoundError as e:
            print(e)
            continue
