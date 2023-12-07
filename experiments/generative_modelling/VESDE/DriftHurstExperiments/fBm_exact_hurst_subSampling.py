import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import reduce_to_fBn, optimise_whittle
from utils.plotting_functions import plot_histogram

if __name__ == "__main__":
    S = 1000
    H = 0.7
    T = 256
    fbn = FractionalBrownianNoise(H=H, rng=np.random.default_rng())

    exact_samples = np.array([fbn.circulant_simulation(N_samples=T).cumsum() for _ in tqdm(range(S), desc="Generating exact samples")]).reshape((S, T))
    approx_true_fBn = reduce_to_fBn(exact_samples, reduce=True)
    even_approx_true_fBn = approx_true_fBn[:, ::2] # Every even index
    odd_approx_true_fBn = approx_true_fBn[:, 1::2] # Every odd index
    assert(approx_true_fBn.shape[0] == even_approx_true_fBn.shape[0] == odd_approx_true_fBn.shape[0])
    assert(even_approx_true_fBn.shape[1] == odd_approx_true_fBn.shape[1])
    hs = []
    even_hs = []
    odd_hs = []
    # Compute Hurst parameters
    for i in tqdm(range(S), desc="Computing Hurst Indexes"):
        hs.append(optimise_whittle(approx_true_fBn, idx=i))
        even_hs.append(optimise_whittle(even_approx_true_fBn, idx=i))
        odd_hs.append(optimise_whittle(odd_approx_true_fBn, idx=i))
    my_hs = [np.array(hs), np.array(even_hs), np.array(odd_hs)]
    titles = ["Exact", "Even Exact", "Odd Exact"]

    for i in range(len(my_hs)):
        fig, ax = plt.subplots()
        ax.axvline(x=H, color="blue", label="True Hurst")
        plot_histogram(my_hs[i], num_bins=150, xlabel="H", ylabel="density",
                       plottitle="Histogram of {} samples' estimated Hurst parameter".format(titles[i]), fig=fig, ax=ax)
        print(my_hs[i].mean())
        print(my_hs[i].std())
        plt.show()

