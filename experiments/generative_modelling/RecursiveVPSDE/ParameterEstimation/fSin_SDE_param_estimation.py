import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ml_collections import ConfigDict

from experiments.generative_modelling.estimate_fSDEs import second_order_estimator, \
    estimate_hurst_from_filter


def estimate_SDEs(config: ConfigDict, sampling_model: str, train_epoch: int) -> None:
    incs = pd.read_csv(
        config.experiment_path.replace("rrrrP", "r4P") + "_{}NEp{}.csv.gzip".format(sampling_model, train_epoch),
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
    # Construct estimator for mean reversion
    if config.param_time == config.max_diff_steps - 1:
        PT = 0
    elif config.param_time == 4600:
        PT = 2
    else:
        PT = 1

    # Plot some marginal distributions for increments
    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
    low = 0
    high = config.ts_length
    for idx in range(0):
        tidx = np.random.randint(low=low, high=high)
        t = time_space[tidx]
        exp_mean = 0.
        exp_var = 1. / config.ts_length
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=paths.shape[0])
        incst = incs[:, tidx]
        plt.hist(incst, bins=150, density=True, label="Simulated")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title(f"Marginal Distributions for Increments at time {t} for epoch {train_epoch}")
        plt.legend()
        plt.show()
        plt.close()

    # Plot some marginal distributions for PATHS
    time_space = np.linspace((1. / config.ts_length), 1., num=config.ts_length)
    low = config.ts_length - 10
    high = config.ts_length
    print(config.data_path)
    true_paths = np.load(config.data_path, allow_pickle=True)
    print(true_paths)
    for idx in range(0):
        tidx = np.random.randint(low=low, high=high)
        t = time_space[tidx]
        tpathst = true_paths[:, tidx]
        pathst = paths[:, tidx]  # Paths[:, 0] corresponds to X_{t_{1}} NOT X_{t_{0}}
        plt.hist(pathst, bins=150, density=True, label="Simulated")
        plt.hist(tpathst, bins=150, density=True, label="Expected")
        plt.title(f"Marginal Distributions for PATHS at time {t} for epoch {train_epoch}")
        plt.legend()
        plt.show()
        plt.close()
    means = pd.read_csv(
        (config.experiment_path + "_{}NEp{}_P{}.csv.gzip".format(sampling_type, train_epoch, PT)).replace(
                        "fSin", "fSinm"),
        compression="gzip").to_numpy()
    means *= (config.ts_length ** (2 * config.hurst))

    # Plot drift as a function of space for a single path
    for _ in range(3):
        idx = np.random.randint(low=0, high=paths.shape[0])
        mean = means[idx, 1:]
        path = paths[idx, :-1]
        paired = zip(path, mean)
        # Sort the pairs based on values of arr1
        sorted_pairs = sorted(paired, key=lambda x: x[0])
        # Separate the pairs back into two arrays
        path, mean = zip(*sorted_pairs)
        mean = np.array(mean)
        path = np.array(path)
        #plt.scatter(path, mean, label="Drift Against State")
        plt.scatter(path, config.mean_rev * np.sin(path), label="Expected Drift Against State")
        plt.title(f"Drift against Path with")
        plt.legend()
        plt.show()
        plt.close()

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_PostMeanScore_fSin_T256_H07_tl_5data import get_config

    config = get_config()
    sampling_models = ["CondAncestral", "CondReverseDiffusion", "CondProbODE"]
    early_stopping = [True]
    for train_epoch in config.max_epochs:
        with open(config.scoreNet_trained_path.replace("/trained_models/", "/training_losses/") + "_loss", 'rb') as f:
            losses = np.array(pickle.load(f))
        assert (losses.shape[0] >= 1)  # max(config.max_epochs))
        T = losses.shape[0]
        plt.plot(np.linspace(1, T + 1, T)[100:10000], (losses.cumsum() / np.arange(1, T + 1))[100:10000])
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("CumMean Per-epoch Training Loss")
        plt.show()
        plt.plot(np.linspace(1, T + 1, T)[100:10000], losses[100:10000])
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Per-epoch Training Loss")
        plt.show()
        print(train_epoch)
        for sampling_model in sampling_models:
            if sampling_model == "CondAncestral":
                sampling_type = "a"
            elif sampling_model == "CondReverseDiffusion":
                sampling_type = "r"
            else:
                sampling_type = "p"
            for early_stop in early_stopping:
                sampling_type = "e" + sampling_type if early_stop else sampling_type
                try:
                    pd.read_csv(
                        config.experiment_path.replace("rrrrP", "r4P") + "_{}NEp{}.csv.gzip".format(sampling_type,
                                                                                                    train_epoch),
                        compression="gzip",
                        index_col=[0, 1]).to_numpy()
                    if config.param_time == config.max_diff_steps - 1:
                        PT = 0
                    elif config.param_time == 4600:
                        PT = 2
                    else:
                        PT = 1
                    estimate_SDEs(config=config, train_epoch=train_epoch, sampling_model=sampling_type)
                    break
                except FileNotFoundError as e:
                    print(e)
                    continue
            break