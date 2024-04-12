import numpy as np
from matplotlib import pyplot as plt

from experiments.generative_modelling.estimate_fSDEs import second_order_estimator, estimate_hurst_from_filter
from utils.math_functions import generate_fOU

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    training_size = 200000
    config = get_config()
    data = generate_fOU(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                        H=config.hurst, mean_rev=config.mean_rev, mean=config.mean, diff=config.diffusion,
                        initial_state=config.initState)
    np.save(config.data_path, data)
    U_a1, U_a2 = second_order_estimator(paths=data, Nsamples=training_size)
    estimate_hurst_from_filter(Ua1=U_a1, Ua2=U_a2)
    time_ax = np.linspace(0, 1, config.ts_length)
    for _ in range(10):
        idx = np.random.randint(low=0, high=training_size)
        path = data[idx, :]
        plt.plot(time_ax, path)
    plt.show()
    plt.close()
