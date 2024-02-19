import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plotting_functions import hurst_estimation

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})
if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    H = config.hurst
    with open(config.scoreNet_trained_path.replace("/trained_models/", "/training_losses/") + "_loss", 'rb') as f:
        losses = np.array(pickle.load(f))
    # Loss file contains losses for same model trained (potentially) sequentially many times
    assert(losses.shape[0] >= max(config.max_epochs))
    T = losses.shape[0]
    plt.plot(np.linspace(1, T+1, T), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Per-epoch Training Loss")
    plt.show()

    plt.plot(np.linspace(1, T+1, T), np.cumsum(losses)/np.arange(1, T+1))
    plt.xlabel("Epoch")
    plt.ylabel("Running Training Loss Mean")
    plt.title("Cumulative Mean for Training Loss")
    plt.show()

    # Now plot Hurst histogram for the generated samples
    for train_epoch in config.max_epochs:
        df = pd.read_csv(config.experiment_path + "_Nepochs{}.csv.gzip".format(train_epoch),compression="gzip", index_col=[0, 1])
        df = df.apply(lambda x: [eval(i.replace("(", "").replace(")","").replace("tensor","")) if type(i) == str else i for i in x]).loc["Final Time Samples"]
        hs = hurst_estimation(df.to_numpy(), sample_type="Final Time Samples at Train Epoch {}".format(train_epoch), isfBm=config.isfBm, true_hurst=config.hurst)