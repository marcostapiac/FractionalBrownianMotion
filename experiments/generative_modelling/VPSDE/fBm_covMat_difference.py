import numpy as np
import pandas as pd

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_fBm_cov, compute_fBn_cov, generate_fBm, generate_fBn
from utils.plotting_functions import plot_diffCov_heatmap

if __name__ == "__main__":
    from configs.VPSDE.fBm_T1024_H07 import get_config

    config = get_config()
    H = config.hurst

    path = config.experiment_path.replace("/results/",
                                          "/results/early_stopping/") + "_Samples_EStop{}_Nepochs{}.csv.gzip".format(
        1, config.max_epochs)
    df = pd.read_csv(path, compression="gzip", index_col=[0, 1])

    # Now onto exact samples for reference
    unitInterval = True if "True_unitIntv" in path else False
    isfBm = True if "False_incs" in path else False

    fbn = FractionalBrownianNoise(H=config.hurst, rng=np.random.default_rng())
    true_cov = compute_fBm_cov(fbn, T=config.timeDim, isUnitInterval=unitInterval) if isfBm else compute_fBn_cov(fbn,
                                                                                                                 T=config.timeDim,
                                                                                                                 isUnitInterval=unitInterval)

    if config.isfBm:
        exact_samples = generate_fBm(H=config.hurst, T=config.timeDim, S=df.index.levshape[1],
                                     isUnitInterval=config.isUnitInterval)
    else:
        exact_samples = generate_fBn(H=config.hurst, T=config.timeDim, S=df.index.levshape[1],
                                     isUnitInterval=config.isUnitInterval)

    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=np.cov(exact_samples.T), annot=False, image_path="")
    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        this_cov = np.cov(df.loc[type].to_numpy().T)
        plot_diffCov_heatmap(true_cov=true_cov, gen_cov=this_cov, annot=False, image_path="")
