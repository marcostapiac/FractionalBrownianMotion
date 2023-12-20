import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import reduce_to_fBn, optimise_whittle
from utils.plotting_functions import plot_histogram

if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Samples_EStop{}_Nepochs{}.csv.gzip".format(
        1,config.max_epochs), compression="gzip", index_col=[0, 1])

    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        print(all(np.sign(df.loc[type].mean(axis=0)) == -1.0))

