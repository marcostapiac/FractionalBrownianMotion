import matplotlib
import ml_collections
import numpy as np
import torch
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from scipy import stats
from tqdm import tqdm

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from utils.math_functions import compute_fBm_cov
from utils.plotting_functions import make_gif, plot_and_save_diffused_fBm_snapshot


def run(perfect_config: ConfigDict) -> None:
    print("!!!! Running started !!!! \n")
    try:
        assert(perfect_config.save_freq <= perfect_config.max_diff_steps)
        assert(perfect_config.save_freq > 0)
        assert(perfect_config.dim1 < perfect_config.timeDim)
        assert(perfect_config.dim2 < perfect_config.timeDim)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    diffusion = VESDEDiffusion(stdMax=perfect_config.std_max, stdMin=perfect_config.std_min)

    gen = FractionalBrownianNoise(perfect_config.hurst, np.random.default_rng())
    fBm_cov = torch.from_numpy(compute_fBm_cov(gen, T=perfect_config.timeDim, isUnitInterval=True)).to(torch.float32)

    data = torch.from_numpy(np.array(
        [gen.circulant_simulation(perfect_config.timeDim).cumsum() for _ in tqdm(range(perfect_config.dataSize))])).to(
        torch.float32)
    dim_pair = torch.Tensor([perfect_config.dim1, perfect_config.dim2]).to(torch.int32)

    # Now choose the dimensions we are interested in
    data = torch.index_select(data, dim=1, index=dim_pair)
    fBm_cov = torch.index_select(torch.index_select(fBm_cov, dim=0, index=dim_pair), dim=1, index=dim_pair)

    ts = np.linspace(0., perfect_config.end_diff_time, num=perfect_config.max_diff_steps)
    folder_path = project_config.ROOT_DIR + "experiments/results/forward_gifs/"
    gif_path = "fBm_dimPair{}_dimPair{}_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_StdMax{:.4e}_StdMin{:.4e}".format(dim_pair[0],
                                                                                                          dim_pair[1],
                                                                                                          perfect_config.hurst,
                                                                                                          perfect_config.timeDim,
                                                                                                          perfect_config.max_diff_steps,
                                                                                                          perfect_config.end_diff_time,
                                                                                                          perfect_config.std_max,
                                                                                                          perfect_config.std_min).replace(
        ".", "")

    for i in tqdm(range(perfect_config.max_diff_steps)):
        eff_time = diffusion.get_eff_times(diff_times=torch.Tensor([ts[i]]))
        xts, _ = diffusion.noising_process(data, eff_time)
        if i % perfect_config.save_freq == 0 or i == (perfect_config.max_diff_steps - 1):
            save_path = folder_path + gif_path + "_diffIndex_{}.png".format(i + 1)
            plot_title = "Forward Diffused Samples with $T={}$ at time {}".format(perfect_config.timeDim,
                                                                                     round((
                                                                                                   i + 1) / perfect_config.max_diff_steps,
                                                                                           5))
            xlabel = "fBm Dimension {}".format(dim_pair[0]+1)
            ylabel = "fBm Dimension {}".format(dim_pair[1]+1)
            cov = eff_time * torch.eye(2) + fBm_cov
            plot_and_save_diffused_fBm_snapshot(samples=xts, cov=cov, save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    config = ml_collections.ConfigDict()
    config.has_cuda = torch.cuda.is_available()
    config.hurst = 0.7
    config.timeDim = 256
    config.max_diff_steps = 5000
    config.end_diff_time = 1
    config.std_max = 90
    config.std_min = 0.01
    config.dim1 = 254
    config.dim2 = 255
    config.dataSize = 5000
    config.save_freq = 10

    # Run experiment
    run(config)
