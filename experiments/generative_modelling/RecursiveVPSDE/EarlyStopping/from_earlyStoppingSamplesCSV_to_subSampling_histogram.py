import pandas as pd

from utils.math_functions import generate_fBm, generate_fBn
from utils.plotting_functions import hurst_estimation

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_2data import get_config
    # _MSeed0
    config = get_config()
    H = config.hurst
    df = pd.read_csv(config.experiment_path.replace("/results/",
                                                    "/results/early_stopping/") + "_Nepochs{}.csv.gzip".format(
     config.max_epochs),
                     compression="gzip", index_col=[0, 1])
    #df = (df.apply(lambda x: [eval(i.replace("(", "").replace(")","").replace("tensor","")) for i in x]))
    # Synthetic samples
    for type in df.index.get_level_values(level=0).unique():
        hurst_estimation(df.loc[type].to_numpy(), sample_type=type, config=config)

    if config.isfBm:
        exact_samples = generate_fBm(H=config.hurst, T=config.timeDim, S=df.index.levshape[1],
                                     isUnitInterval=config.isUnitInterval)
    else:
        exact_samples = generate_fBn(H=config.hurst, T=config.timeDim, S=df.index.levshape[1],
                                     isUnitInterval=config.isUnitInterval)

    hurst_estimation(exact_samples, sample_type="exact", config=config)
