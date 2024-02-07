import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils.plotting_functions import hurst_estimation

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07 import get_config
    # _MSeed0
    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path + "_Nepochs{}_MSeed0.csv.gzip".format(
     config.max_epochs),
                     compression="gzip", index_col=[0, 1])
    paths = df.values
    for path in paths:
        plt.plot(np.linspace(1, df.shape[1]+1, df.shape[1]),path)
    plt.xlabel("Time")
    plt.ylabel("Path Value")
    plt.legend()
    plt.title("Reproducible Paths with Same Seed")
    plt.show()

    hurst_estimation(df.loc["Final Time Samples"].to_numpy(), sample_type="Final Time Samples", config=config)
