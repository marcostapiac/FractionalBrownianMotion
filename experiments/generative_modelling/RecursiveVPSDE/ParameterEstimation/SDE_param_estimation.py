import pandas as pd
import numpy as np
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from experiments.generative_modelling.estimate_fSDEs import estimate_fSDE_from_true, second_order_estimator, \
    estimate_hurst_from_filter


def estimate_SDEs(config:ConfigDict, train_epoch:int)->None:
    paths = pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip", index_col=[0])
    # Estimate Hurst indices from paths
    U_a1, U_a2 = second_order_estimator(paths=paths.to_numpy(), Nsamples=N)
    path_hs = estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=0)
    # Estimate Hurst indices from true
    true_hs = estimate_fSDE_from_true(config=config)
    # Remove any path with hurst index outside of range
    delidx = np.array([(min(true_hs) <= path_hs[i] <= max(true_hs)) for i in range(path_hs)])
    paths = paths.to_numpy()[delidx,:]
    means = pd.read_csv((config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch)).replace("fOU", "fOUm"), compression="gzip", index_col=[0])
    means = means.to_numpy()[delidx,:]
    vars = pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip", index_col=[0])
    vars = vars.to_numpy()[delidx,:]
    # Now use EM approximation to compute the instantaneous parameters at every point in time and space
    vars /= config.ts_length
    means = means + paths # TODO: Check if this is correct (ti-1 with ti-1)
    means /= config.ts_length
    # Now plot histograms across time and space of estimates
    plt.hist(means, bins=150, density=True)
    plt.title("Instantaneous Drift Estimates")
    plt.show()
    plt.close()
    plt.hist(vars, bins=150, density=True)
    plt.title("Instantaneous Volatility Estimates")
    plt.show()
    plt.close()

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config
    config = get_config()
    for train_epoch in max(config.train_epochs):
        estimate_SDEs(config=config, train_epoch=train_epoch)