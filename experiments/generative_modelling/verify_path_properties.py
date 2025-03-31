import matplotlib.pyplot as plt
import numpy as np

from experiments.generative_modelling.estimate_fSDEs import estimate_fSDE_from_true


def exact_hurst():
    from configs.RecursiveVPSDE.recursive_LSTM_fOU_T256_H07_tl_5data import get_config
    config = get_config()
    # Check generated paths have correct Hurst
    estimate_fSDE_from_true(config=config)
    # Check histogram for each time
    data = np.load(config.data_path, allow_pickle=True)
    # data = pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(480), compression="gzip", index_col=[0,1]).to_numpy()
    time_space = np.linspace(0, 1. + (1. / config.ts_length), num=config.ts_length + 1)[1:]
    sidx = 254
    for tidx in range(sidx, config.ts_length):
        t = time_space[tidx]
        expmeanrev = np.exp(-config.mean_rev * t)
        exp_mean = config.mean * (1. - expmeanrev)
        exp_mean += config.initState * expmeanrev
        exp_var = np.power(config.diffusion, 2)
        exp_var /= (2 * config.mean_rev)
        exp_var *= 1. - np.power(expmeanrev, 2)
        exp_rvs = np.random.normal(loc=exp_mean, scale=np.sqrt(exp_var), size=data.shape[0])
        pathst = data[:, tidx]
        plt.hist(pathst, bins=150, density=True, label="True")
        plt.hist(exp_rvs, bins=150, density=True, label="Expected")
        plt.title("Marginal Distributions at time {}".format(t))
        plt.legend()
        plt.show()
        plt.close()
    # Construct estimator for mean reversion
    augdata = np.concatenate([np.zeros(shape=(data.shape[0], 1)), data], axis=1)
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


if __name__ == "__main__":
    # exact_hurst()
    H = 0.5
    l1 = (2 * H - 1) / (2 - 2 * H)
    l2 = 1. / (4. - 4 * H)
    alpha = max(l1, max(l2, 1))
    print(alpha)
    assert (alpha == 1)
    exact_hurst()
