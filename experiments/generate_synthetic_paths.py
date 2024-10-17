import numpy as np
import torch
from matplotlib import pyplot as plt

from configs import project_config
from experiments.generative_modelling.estimate_fSDEs import second_order_estimator, estimate_hurst_from_filter
from utils.math_functions import generate_fSin, generate_fOU

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_PostMeanScore_fOU_T256_H05_tl_5data import get_config

    training_size = 200000
    config = get_config()
    print(config.hurst, config.mean_rev, config.mean)
    data = generate_fOU(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                         H=config.hurst, mean_rev=config.mean_rev, diff=config.diffusion,
                         initial_state=config.initState, mean=config.mean)
    np.save(config.data_path, data)
    #data = np.load(config.data_path)
    #data = torch.load(project_config.ROOT_DIR + f"experiments/results/TSPM_DriftEvalExp_{12920}Nep_{config.loss_factor}LFactor_prevPaths").numpy()

    U_a1, U_a2 = second_order_estimator(paths=data, Nsamples=training_size)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2, epoch=0)

    augdata = data
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

    time_ax = np.linspace(0, 1, config.ts_length)
    for _ in range(10):
        idx = np.random.randint(low=0, high=training_size)
        path = data[idx, :]
        plt.plot(time_ax, path)
    plt.show()
    plt.close()
