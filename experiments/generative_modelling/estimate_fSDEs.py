import numpy as np
from ml_collections import ConfigDict


def estimate_fSDEs(config: ConfigDict):
    fOU = np.load(config.data_path, allow_pickle=True)
    # We want to construct first an estimator for H based on sample paths


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_fOU_T256_H07_tl_5data import get_config

    config = get_config()
    estimate_fSDEs(config=config)
