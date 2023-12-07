import numpy as np
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_fBm_cov
from utils.plotting_functions import plot_diffCov_heatmap
import pandas as pd


if __name__ == "__main__":
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()
    H = config.hurst

    df = pd.read_csv(config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Samples_EarlyStoppingExperiment_Nepochs{}.csv.gzip".format(
        config.max_epochs),
                     compression="gzip", index_col=[0, 1])
    # Now onto exact samples for reference
    fbn = FractionalBrownianNoise(H=config.hurst, rng=np.random.default_rng())
    true_cov = compute_fBm_cov(fbn,T=config.timeDim, isUnitInterval=True)

    S = df.index.levshape[1]
    fbm_samples = np.array(
        [fbn.circulant_simulation(N_samples=config.timeDim).cumsum() for _ in tqdm(range(S))]).reshape(
        (S, config.timeDim))
    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=np.cov(fbm_samples.T), annot=False, image_path="")
    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        this_cov = np.cov(df.loc[type].to_numpy().T)
        plot_diffCov_heatmap(true_cov=true_cov, gen_cov=this_cov, annot=False, image_path="")