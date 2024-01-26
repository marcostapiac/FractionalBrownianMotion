import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from utils.plotting_functions import plot_efftimes


def run(perfect_config: ConfigDict) -> None:
    diffusion = VPSDEDiffusion(beta_max=perfect_config.beta_max, beta_min=perfect_config.beta_min)
    ts = np.linspace(config.sample_eps, config.end_diff_time, config.max_diff_steps)
    efftimes = diffusion.get_eff_times(torch.Tensor(ts)).numpy()
    plot_efftimes(ts, efftimes)
    plot_efftimes(ts[:10], efftimes[:10])
    plot_efftimes(ts[:3], efftimes[:3])
    print(efftimes[199])


if __name__ == "__main__":
    # Data parameters
    config = ml_collections.ConfigDict()
    config.has_cuda = torch.cuda.is_available()
    config.hurst = 0.7
    config.timeDim = 1024
    config.max_diff_steps = 10000
    config.end_diff_time = 1
    config.beta_max = 20
    config.beta_min = 0.0001
    config.sample_eps = 1e-4

    # Run experiment
    run(config)
