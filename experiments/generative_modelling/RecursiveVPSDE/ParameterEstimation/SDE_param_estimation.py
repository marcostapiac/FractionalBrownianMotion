import time

import pandas as pd
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from tqdm import tqdm

from experiments.generative_modelling.estimate_fSDEs import estimate_fSDE_from_true, second_order_estimator, \
    estimate_hurst_from_filter


def estimate_SDEs(config: ConfigDict, train_epoch: int) -> None:
    incs = pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip",
                       index_col=[0, 1]).to_numpy()
    means = -1*pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUm"),
                        compression="gzip", index_col=[0, 1]).to_numpy()
    vars = pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUv"), compression="gzip",
                       index_col=[0, 1]).to_numpy()
    paths = incs.cumsum(axis=1)
    # Now use EM approximation to compute the instantaneous parameters at every point in time and space
    vars *= config.ts_length
    means *= config.ts_length
    # Construct estimator for mean reversion
    augdata = np.concatenate([np.zeros(shape=(paths.shape[0], 1)), paths], axis=1)
    denom = np.sum(np.power(augdata[:, :-1], 2), axis=1) * (1. / config.ts_length)
    diffs = np.diff(augdata, n=1, axis=1)
    num = np.sum(augdata[:, :-1] * diffs, axis=1)
    mean_revs = -num / denom
    plt.hist(mean_revs, bins=150, density=True)
    plt.vlines(x=config.mean_rev, ymin=0, ymax=0.25, label="", color='b')
    plt.title("Mean Reversion estimator for synthetic paths")
    plt.legend()
    plt.show()
    plt.close()
    print(np.mean(mean_revs), np.std(mean_revs))
    print(np.mean(vars), np.std(vars))
    # Construct estimator for mean reversion using true paths
    true_data = np.load(config.data_path, allow_pickle=True)
    augdata = np.concatenate([np.zeros(shape=(true_data.shape[0], 1)), true_data], axis=1)
    denom = np.sum(np.power(augdata[:, :-1], 2), axis=1) * (1. / config.ts_length)
    diffs = np.diff(augdata, n=1, axis=1)
    num = np.sum(augdata[:, :-1] * diffs, axis=1)
    mean_revs = -num / denom
    plt.hist(mean_revs, bins=150, density=True)
    plt.vlines(x=config.mean_rev, ymin=0, ymax=0.25, label="", color='b')
    plt.title("Mean Reversion estimator for true paths")
    plt.legend()
    plt.show()
    plt.close()
    # Compare marginal distributions
    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
    sidx = 253
    for tidx in range(sidx, config.ts_length):
        t = time_space[tidx]
        expmeanrev = np.exp(-config.mean_rev * t)
        exp_mean = config.mean * (1. - expmeanrev)
        exp_mean += config.initState * expmeanrev
        exp_var = np.power(config.diffusion, 2)
        exp_var /= (2 * config.mean_rev)
        exp_var *= 1. - np.power(expmeanrev, 2)
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=paths.shape[0])
        pathst = paths[:, tidx] # Paths[:, 0] corresponds to X_{t_{1}} NOT X_{t_{0}}
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title("Marginal Distributions at time {}".format(t))
        plt.legend()
        plt.show()
        plt.close()
    # Estimate Hurst indices from paths
    U_a1, U_a2 = second_order_estimator(paths=paths, Nsamples=paths.shape[0])
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=train_epoch).flatten()
    for _ in range(1000):
        idx = np.random.randint(0, paths.shape[0])
        plt.plot(np.linspace(0, 1, config.ts_length), paths[idx, :])
    plt.show()
    plt.close()
    # Estimate Hurst indices from true
    estimate_fSDE_from_true(config=config).flatten()
    # Now plot histograms across time and space of estimates
    for t in range(1, config.ts_length):
        varst = vars[:, t]
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
        train_epoch = 303
        print(train_epoch)
        estimate_SDEs(config=config, train_epoch=train_epoch)
