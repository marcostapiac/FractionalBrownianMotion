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
    means = -1 * pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUm"),
                             compression="gzip", index_col=[0, 1]).to_numpy()
    vars = pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUv"),
                       compression="gzip",
                       index_col=[0, 1]).to_numpy()
    paths = incs.cumsum(axis=1)
    for _ in range(paths.shape[0]):
        plt.plot(np.linspace(0, 1, config.ts_length), paths[_, :])
    plt.show()
    plt.close()
    vars *= config.ts_length
    means *= config.ts_length
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
    plt.title("Mean Reversion Linear Regression Estimates")
    plt.show()
    plt.close()
    path_ids = np.arange(paths.shape[0])[np.abs(np.array(mean_revs))>10]
    faulty_paths = paths[np.abs(np.array(mean_revs))>10, :]
    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
    for _ in range(faulty_paths.shape[0]):
        plt.plot(time_space, faulty_paths)
    plt.show()
    plt.close()
    # Now plot histograms across time and space of estimates
    for i in range(3):
        t = np.random.randint(low=1, high=config.ts_length)
        meant = means[:, t]
        varst = vars[:, t]
        pathst = paths[:, t - 1]
        plt.scatter(pathst, meant)
        plt.plot(pathst, -config.mean_rev*pathst, color="blue")
        plt.ylim((-config.mean_rev*max(pathst), -config.mean_rev*min(pathst)))
        plt.title(f"Drift Function at time {t + 1} against state value")
        plt.show()
        plt.close()
        time.sleep(0.5)
        plt.scatter(pathst, varst)
        plt.plot(pathst, [config.diffusion]*len(pathst), color="blue")
        plt.ylim((-1, 2))
        plt.title(f"Diffusion Function at time {t + 1} against state value")
        plt.show()
        plt.close()
        time.sleep(0.5)
    plt.hist(vars.flatten(), bins=150, density=True)
    plt.vlines(x=config.diffusion, ymin=0, ymax=0.4, color="blue")
    plt.xlim((-2, 50))
    plt.title("Instantaneous Vol Estimates")
    plt.show()
    plt.close()  # Compare marginal distributions


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    for train_epoch in config.max_epochs:
        try:
            pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip",
                        index_col=[0, 1]).to_numpy()
            print(train_epoch)
            train_epoch = 2920
            estimate_SDEs(config=config, train_epoch=train_epoch)
        except FileNotFoundError:
            continue
