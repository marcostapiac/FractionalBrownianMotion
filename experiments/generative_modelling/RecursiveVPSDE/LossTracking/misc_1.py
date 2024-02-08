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
    data = np.load("/Users/marcos/GitHubRepos/FractionalBrownianMotion/data/fBn_samples_H07_T256.npy")
    print(data.shape)
    data = data[::-1,:]

    for i in range(0, data.shape[0], 20000):
        cd = data[i:i+20000,:]
        hurst_estimation(cd, sample_type="Final Time Samples", isfBm=False, true_hurst=0.7)
