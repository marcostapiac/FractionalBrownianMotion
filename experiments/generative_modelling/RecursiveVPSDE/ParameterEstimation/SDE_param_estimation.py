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
    # TODO: Note -1 because of bug in original code (only for H0U5 case)
    PT = 0 if config.param_time == config.max_diff_steps - 1 else 1
    means = pd.read_csv((config.experiment_path + "_NEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU", "fOUm"),
                             compression="gzip", index_col=[0, 1]).to_numpy()
    vars = pd.read_csv((config.experiment_path + "_NEp{}_PT{}.csv.gzip".format(train_epoch, PT)).replace("fOU", "fOUv"),
                       compression="gzip",
                       index_col=[0, 1]).to_numpy()
    paths = incs.cumsum(axis=1)
    #for _ in range(paths.shape[0]):
    #    plt.plot(np.linspace(0, 1, config.ts_length), paths[_, :])
    #plt.show()
    #plt.close()
    # Estimate Hurst indices from paths
    #U_a1, U_a2 = second_order_estimator(paths=paths, Nsamples=paths.shape[0])
    #estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=train_epoch).flatten()
    # Estimate Hurst indices from true
    #estimate_fSDE_from_true(config=config).flatten()
    # Now use EM approximation to compute the instantaneous parameters at every point in time and space
    #vars *= (config.ts_length**(2*config.hurst))
    means *= (config.ts_length**(2*config.hurst))
    M = means.shape[0]
    N = means.shape[1] - 1
    mean_revs = []
    means = means[:, 1:]
    for pathidx in tqdm(range(M)):
        ys = means[pathidx, :].flatten().reshape((N, 1))
        designmat = paths[pathidx, :-1].flatten().reshape((N, 1))
        meanrev = -np.linalg.solve(designmat.T @ designmat, designmat.T @ ys)
        mean_revs.append(float(meanrev))
    plt.hist(mean_revs, bins=150, density=True)
    plt.vlines(x=config.mean_rev, ymin=0, ymax=0.4, color="blue")
    plt.xlim((-20, 20))
    plt.title(f"Mean Reversion Linear Regression Estimates for epoch {train_epoch}")
    plt.show()
    plt.close()
    plt.hist(vars.flatten(), bins=150, density=True)
    plt.vlines(x=config.diffusion, ymin=0, ymax=0.4, color="blue")
    plt.xlim((-2, 50))
    plt.title(f"Instantaneous Vol Estimates for epoch {train_epoch}")
    plt.show()
    plt.close()    # Compare marginal distributions
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
        pathst = paths[:, tidx] # Paths[:, 0] corresponds to X_{t_{1}} NOT X_{t_{0}}
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title(f"Marginal Distributions at time {t+1} for epoch {train_epoch}")
        plt.legend()
        plt.show()
        plt.close()

    for idx in range(paths.shape[0]):
        path = paths[idx, :-1]
        meanp = means[idx, :]
        plt.scatter(path, meanp)
        plt.plot(path, -config.mean_rev*path, color="blue")
        plt.show()
        plt.close()
        time.sleep(1)
    # Now plot histograms across time and space of estimates
    for i in range(3):
        t = np.random.randint(low=1, high=config.ts_length-1)
        meant = means[:, t]
        varst = vars[:, t]
        pathst = paths[:, t - 1]
        plt.scatter(pathst, meant)
        plt.plot(pathst, -config.mean_rev*pathst, color="blue")
        plt.ylim((-config.mean_rev*max(pathst), -config.mean_rev*min(pathst)))
        plt.title(f"Drift Function at time index {t+1} against state value for epoch {train_epoch}")
        plt.close()
        time.sleep(0.5)
        plt.scatter(pathst, meant)
        plt.plot(pathst, -config.mean_rev * pathst, color="blue")
        plt.title(f"Drift Function at time index {t + 1} against state value for epoch {train_epoch}")
        plt.show()
        plt.close()
        time.sleep(0.5)
        plt.scatter(pathst, varst, s=0.1)
        plt.plot(pathst, [config.diffusion]*len(pathst), color="blue")
        plt.ylim((-1, 2))
        plt.title(f"Diffusion Function at time index {t + 1} against state value for epoch {train_epoch}")
        plt.show()
        plt.close()
        time.sleep(0.5)
        plt.scatter(pathst, varst, s=0.1)
        plt.plot(pathst, [config.diffusion] * len(pathst), color="blue")
        plt.title(f"Diffusion Function at time index {t + 1} against state value for epoch {train_epoch}")
        plt.show()
        plt.close()
        time.sleep(0.5)


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    for train_epoch in config.max_epochs:
        try:
            pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip",
                        index_col=[0, 1]).to_numpy()
            train_epoch = 2920
            print(train_epoch)
            print(config.param_time)
            config.param_time = 100
            estimate_SDEs(config=config, train_epoch=train_epoch)
        except FileNotFoundError as e:
            print(e)
            continue
