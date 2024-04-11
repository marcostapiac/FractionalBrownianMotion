from matplotlib import pyplot as plt

from utils.math_functions import generate_fOU
import numpy as np
if __name__=="__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config
    training_size = 200000
    config = get_config()
    data = generate_fOU(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                        H=config.hurst, mean_rev=config.mean_rev, mean=config.mean, diff=config.diffusion,
                        initial_state=config.initState)
    np.save(config.data_path, data)
    time_ax = np.linspace(0,1, config.ts_length)
    for _ in range(10):
        idx = np.random.randint(low=0, high=training_size)
        path = data[idx, :]
        plt.plot(time_ax, path)
    plt.show()
    plt.close()