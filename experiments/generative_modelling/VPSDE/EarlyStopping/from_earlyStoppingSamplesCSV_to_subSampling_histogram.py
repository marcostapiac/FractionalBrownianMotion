import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.math_functions import reduce_to_fBn, optimise_whittle
from utils.plotting_functions import plot_histogram

if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config
    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path + "_Samples_EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(config.max_epochs),
                     compression="gzip", index_col=[0,1])
    for sample_type in df.index.get_level_values(level=0).unique():
        approx_fBn = reduce_to_fBn(df.loc[sample_type].to_numpy(), reduce=True)
        if sample_type == "Synthetic": sample_type = "Early Stop Synthetic"
        even_approx_fBn = approx_fBn[:, ::8]  # Every even eigth index
        odd_approx_fBn = approx_fBn[:, 1::8]  # Every odd eigth index
        assert (approx_fBn.shape[0] == even_approx_fBn.shape[0] == odd_approx_fBn.shape[0])
        assert (even_approx_fBn.shape[1] == odd_approx_fBn.shape[1])
        hs = []
        even_hs = []
        odd_hs = []
        S  = approx_fBn.shape[0]
        # Compute Hurst parameters
        for i in tqdm(range(S), desc="Computing Hurst Indexes"):
            hs.append(optimise_whittle(approx_fBn, idx=i))
            even_hs.append(optimise_whittle(even_approx_fBn, idx=i))
            odd_hs.append(optimise_whittle(odd_approx_fBn, idx=i))

        my_hs = [np.array(hs), np.array(even_hs), np.array(odd_hs)]
        titles = ["Exact", "Even Exact", "Odd Exact"]

        for i in range(len(my_hs)):
            fig, ax = plt.subplots()
            ax.axvline(x=H, color="blue", label="True Hurst")
            plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                           plottitle="Histogram of {} {} samples' estimated Hurst parameter".format(titles[i], sample_type), fig=fig, ax=ax)
            print(my_hs[i].mean())
            print(my_hs[i].std())
            plt.show()
