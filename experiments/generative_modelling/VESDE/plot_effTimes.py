import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from utils.plotting_functions import plot_efftimes


def run(perfect_config: ConfigDict) -> None:
    diffusion = VESDEDiffusion(stdMax=perfect_config.std_max, stdMin=perfect_config.std_min)
    ts = np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)
    efftimes = diffusion.get_eff_times(torch.Tensor(ts)).numpy()
    plot_efftimes(ts, efftimes)
    plot_efftimes(ts[:10], efftimes[:10])
    plot_efftimes(ts[:2], efftimes[:2])
    print(efftimes[:2])


if __name__ == "__main__":
    # Data parameters
    config = ml_collections.ConfigDict()
    config.has_cuda = torch.cuda.is_available()
    config.hurst = 0.7
    config.timeDim = 256
    config.max_diff_steps = 20000
    config.end_diff_time = 1
    config.std_max = 90
    config.std_min = 0.01
    config.sample_eps = 1e-5

    # Run experiment
    run(config)
