import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r"\usepackage{amsmath}"
})
if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_2data import get_config
    config = get_config()
    H = config.hurst
    losses = np.array(pd.read_pickle(config.scoreNet_trained_path.replace("/trained_models/", "/training_losses/") + "_loss_Nepochs{}".format(config.max_epochs)))
    # Losses has length Nepochs x Batch_size
    bsz = int(losses.shape[0]/config.max_epochs)
    per_epoch_loss = np.array([np.sum(losses[i:i+bsz])/bsz for i in range(0, losses.shape[0],bsz)])
    T = per_epoch_loss.shape[0]

    plt.plot(np.linspace(1, T+1, T), per_epoch_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Per-epoch Training Loss")
    plt.show()

    plt.plot(np.linspace(1, T+1, T), np.cumsum(per_epoch_loss)/np.arange(1, T+1))
    plt.xlabel("Epoch")
    plt.ylabel("Running Training Loss Mean")
    plt.title("Cumulative Mean for Training Loss")
    plt.show()
