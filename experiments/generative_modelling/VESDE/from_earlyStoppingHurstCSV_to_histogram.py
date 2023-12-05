import matplotlib.pyplot as plt
import pandas as pd

from utils.plotting_functions import plot_histogram

if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    df = pd.read_csv(config.experiment_path + "EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(config.max_epochs),
                     compression="gzip", index_col=[0])
    title = ["exact", "synthetic", "no stop synthetic"]
    for i in range(df.shape[1]):
        fig, ax = plt.subplots()
        ax.axvline(x=config.hurst, color="blue", label="True Hurst")
        plot_histogram(df.iloc[:, i], num_bins=150, xlabel="H", ylabel="density",
                       plottitle="Histogram of {} samples' estimated Hurst parameter".format(title[i]), fig=fig, ax=ax)
        print(df.iloc[:, i].mean())
        print(df.iloc[:, i].std())
        plt.show()
