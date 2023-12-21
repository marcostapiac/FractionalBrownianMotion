import numpy as np
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_fBm_cov, compute_fBn_cov
from utils.plotting_functions import plot_diffCov_heatmap
import pandas as pd


if __name__ == "__main__":
    from configs.VPSDE.increment_fBm_T256_H07 import get_config

    config = get_config()
    H = config.hurst

    path = config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Samples_EStop{}_Nepochs{}.csv.gzip".format(
        1,config.max_epochs)
    df = pd.read_csv(path, compression="gzip", index_col=[0, 1])

    # Now onto exact samples for reference
    unitInterval = True if "True_unitIntv" in path else False
    isfBm = True if "False_incs" in path else False
    print(unitInterval, isfBm)

    fbn = FractionalBrownianNoise(H=config.hurst, rng=np.random.default_rng())
    true_cov = compute_fBm_cov(fbn,T=config.timeDim, isUnitInterval=unitInterval) if isfBm else compute_fBn_cov(fbn,T=config.timeDim, isUnitInterval=unitInterval)

    S = df.index.levshape[1]
    exact_samples = []
    for _ in tqdm(range(S)):
        tmp = fbn.circulant_simulation(N_samples=config.timeDim, scaleUnitInterval=unitInterval)
        if isfBm: tmp = tmp.cumsum()
        exact_samples.append(tmp)

    exact_samples = np.array(exact_samples).reshape((S, config.timeDim))

    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=np.cov(exact_samples.T), annot=False, image_path="")
    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        this_cov = np.cov(df.loc[type].to_numpy().T)
        plot_diffCov_heatmap(true_cov=true_cov, gen_cov=this_cov, annot=False, image_path="")