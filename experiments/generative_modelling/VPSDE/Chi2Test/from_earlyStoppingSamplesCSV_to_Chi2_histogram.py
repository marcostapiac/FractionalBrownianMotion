import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from ml_collections import ConfigDict
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import reduce_to_fBn, optimise_whittle, chiSquared_test
from utils.plotting_functions import plot_histogram


def one_model_run(config:ConfigDict,fBm_samples: np.ndarray, sample_type: str):
    approx_fBn = reduce_to_fBn(fBm_samples, reduce=True)
    if sample_type == "Synthetic": sample_type = "Early Stop 1"
    even_approx_fBn = approx_fBn[:, ::2]  # Every even index

    print(sample_type)
    # Compute Hurst parameters
    low, chi2s, high = chiSquared_test(T=config.timeDim, H=config.hurst,samples=approx_fBn,isUnitInterval=config.unitInterval)
    low_even, even_chi2s, high_even = chiSquared_test(T=config.timeDim//2, H=config.hurst,samples=even_approx_fBn,isUnitInterval=config.unitInterval)
    my_chi2s = [np.array(chi2s), np.array(even_chi2s)]
    my_bounds = [(low, high), (low_even, high_even)]
    titles = ["All", "Even"]
    dfs = [config.timeDim-1, config.timeDim//2 -1]
    for i in range(len(my_chi2s)):
        fig, ax = plt.subplots()
        ax.axvline(x=my_bounds[i][0], color="blue", label="Lower Bound")
        ax.axvline(x=my_bounds[i][1], color="blue", label="Upper Bound")
        xlinspace = np.linspace(scipy.stats.chi2.ppf(0.0001, dfs[i]), scipy.stats.chi2.ppf(0.9999, dfs[i]), 1000)
        pdfvals = scipy.stats.chi2.pdf(xlinspace, df=dfs[i])
        plot_histogram(my_chi2s[i], pdf_vals=pdfvals, xlinspace=xlinspace, num_bins=200,
                       xlabel="Chi2 Statistic",
                       ylabel="density", plotlabel="Chi2 with {} DoF".format(dfs[i]),
                       plottitle="Histogram of {} {} samples' Chi2 Test Statistic".format(titles[i], sample_type),
                       fig=fig, ax=ax)
        ax.set_xlim(my_bounds[i][0]/10, 10*my_bounds[i][1])
        plt.show()


if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Samples_EarlyStopping{}_Nepochs{}.csv.gzip".format(
        1,config.max_epochs),
                     compression="gzip", index_col=[0, 1])
    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        one_model_run(config=config, fBm_samples=df.loc[type].to_numpy(), sample_type=type)
