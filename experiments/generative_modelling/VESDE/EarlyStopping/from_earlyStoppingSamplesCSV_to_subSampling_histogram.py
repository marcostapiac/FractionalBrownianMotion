import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import reduce_to_fBn, optimise_whittle
from utils.plotting_functions import plot_histogram


def one_model_run(fBm_samples: np.ndarray, sample_type: str):
    approx_fBn = reduce_to_fBn(fBm_samples, reduce=True)
    if sample_type == "Synthetic": sample_type = "Early Stop"
    even_approx_fBn = approx_fBn[:, ::2]  # Every even index

    print(sample_type)
    hs = []
    even_hs = []
    S = approx_fBn.shape[0]
    # Compute Hurst parameters
    for i in tqdm(range(S), desc="Computing Hurst Indexes"):
        hs.append(optimise_whittle(approx_fBn, idx=i))
        even_hs.append(optimise_whittle(even_approx_fBn, idx=i))

    my_hs = [np.array(hs), np.array(even_hs)]
    titles = ["All", "Even"]

    for i in range(len(my_hs)):
        fig, ax = plt.subplots()
        ax.axvline(x=H, color="blue", label="True Hurst")
        plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                       plottitle="Histogram of {} {} samples' estimated Hurst parameter".format(titles[i], sample_type),
                       fig=fig, ax=ax)
        mean, std = my_hs[i].mean(), my_hs[i].std()
        print(mean)
        print(std)
        plt.show()
        # Repeat with constrained axis
        fig, ax = plt.subplots()
        ax.axvline(x=H, color="blue", label="True Hurst")
        plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                       plottitle="Constrained hist of {} {} samples' estimated Hurst parameter".format(titles[i], sample_type),
                       fig=fig, ax=ax)
        ax.set_xlim(mean - 5 * std, mean + 5 * std)
        plt.show()


if __name__ == "__main__":
    from configs.VESDE.fBm_T256_H07 import get_config

    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Samples_EarlyStopping{}_Nepochs{}.csv.gzip".format(
        1,config.max_epochs), compression="gzip", index_col=[0, 1])

    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        one_model_run(df.loc[type].to_numpy(), sample_type=type)

    # Now onto exact samples for reference
    fbn = FractionalBrownianNoise(H=config.hurst, rng=np.random.default_rng())
    S = df.index.levshape[1]
    exact_samples = np.array(
        [fbn.circulant_simulation(N_samples=config.timeDim).cumsum() for _ in tqdm(range(S))]).reshape(
        (S, config.timeDim))
    one_model_run(fBm_samples=exact_samples, sample_type="exact")

