import time

import matplotlib.pyplot as plt
import numpy as np

from experiments.generative_modelling.estimate_fSDEs import estimate_fSDEs, estimate_fSDE_from_true


def exact_hurst():
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config
    config = get_config()
    # Check generated paths have correct Hurst
    estimate_fSDE_from_true(config=config)
    # Check histogram for each time
    data = np.load(config.data_path, allow_pickle=True)
    time_space = np.linspace(0,1.+(1./config.ts_length),num=config.ts_length+1)[1:]
    for tidx in range(config.ts_length):
        t = time_space[tidx]
        expmeanrev = np.exp(-config.mean_rev*t)
        exp_mean = config.mean*(1.-expmeanrev)
        exp_mean += config.initState*expmeanrev
        exp_var = np.power(config.diffusion, 2)
        exp_var /= (2*config.mean_rev)
        exp_var *= 1.-np.power(expmeanrev, 2)
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var),size=data.shape[0])
        pathst = data[:, tidx]
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title("Marginal Distributions at time {}".format(t))
        plt.legend()
        plt.show()
        plt.close()
        time.sleep(.1)





if __name__ == "__main__":
    # exact_hurst()
    H = 0.5
    l1 = (2 * H - 1) / (2 - 2 * H)
    l2 = 1. / (4. - 4 * H)
    alpha = max(l1, max(l2, 1))
    print(alpha)
    assert(alpha == 1)
    exact_hurst()
