import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from utils.math_functions import compute_fBm_cov, compute_fBn_cov
from utils.plotting_functions import make_gif, plot_and_save_diffused_fBm_snapshot


def run(forward_config: ConfigDict) -> None:
    print("!!!! Running started !!!! \n")
    try:
        assert (forward_config.gif_save_freq <= forward_config.max_diff_steps)
        assert (forward_config.gif_save_freq > 0)
        assert (forward_config.dim1 < forward_config.timeDim)
        assert (forward_config.dim2 < forward_config.timeDim)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    diffusion = VESDEDiffusion(stdMax=forward_config.std_max, stdMin=forward_config.std_min)
    dim_pair = torch.Tensor([forward_config.dim1, forward_config.dim2]).to(torch.int32)

    gen = FractionalBrownianNoise(forward_config.hurst, np.random.default_rng())
    if forward_config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(gen, T=forward_config.timeDim, isUnitInterval=forward_config.isUnitInterval)).to(
            torch.float32)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(gen, T=forward_config.timeDim, isUnitInterval=forward_config.isUnitInterval)).to(
            torch.float32)

    org_data = []
    for _ in tqdm(range(forward_config.dataSize)):
        tmp = gen.circulant_simulation(forward_config.timeDim, scaleUnitInterval=forward_config.isUnitInterval)
        if forward_config.isfBm: tmp = tmp.cumsum()
        org_data.append(tmp)

    org_data = torch.from_numpy(np.array(org_data)).to(torch.float32)

    # Now choose the dimensions we are interested in
    org_data = torch.index_select(org_data, dim=1, index=dim_pair)
    data_cov = torch.index_select(torch.index_select(data_cov, dim=0, index=dim_pair), dim=1, index=dim_pair)

    ts = np.linspace(0., forward_config.end_diff_time, num=forward_config.max_diff_steps)
    folder_path = project_config.ROOT_DIR + "experiments/results/forward_gifs/"
    gif_path = "{}_incs_{}_unitIntv_fBm_dimPair{}_dimPair{}_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_StdMax{:.4e}_StdMin{:.4e}".format(
        not forward_config.isfBm, forward_config.isUnitInterval, dim_pair[0],
        dim_pair[1],
        forward_config.hurst,
        forward_config.timeDim,
        forward_config.max_diff_steps,
        forward_config.end_diff_time,
        forward_config.std_max,
        forward_config.std_min).replace(
        ".", "")

    for i in tqdm(range(forward_config.max_diff_steps)):
        eff_time = diffusion.get_eff_times(diff_times=torch.Tensor([ts[i]]))
        xts, _ = diffusion.noising_process(org_data, eff_time)
        if i % forward_config.gif_save_freq == 0 or i == (forward_config.max_diff_steps - 1):
            save_path = folder_path + gif_path + "_diffIndex_{}.png".format(i + 1)
            plot_title = "Forward Diffused Samples with $T={}$ at time {}".format(forward_config.timeDim,
                                                                                  round((
                                                                                                i + 1) / forward_config.max_diff_steps,
                                                                                        5))
            xlabel = "fBm Dimension {}".format(dim_pair[0] + 1)
            ylabel = "fBm Dimension {}".format(dim_pair[1] + 1)
            if i == (forward_config.max_diff_steps - 1):
                # We compare the final forward time samples with the p_{noise} we sample from.
                cov = eff_time * torch.eye(2)
            else:
                cov = eff_time * torch.eye(2) + data_cov
            plot_and_save_diffused_fBm_snapshot(samples=xts, cov=cov, save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    forward_config = ml_collections.ConfigDict()
    forward_config.has_cuda = torch.cuda.is_available()
    forward_config.hurst = 0.7
    forward_config.timeDim = 256
    forward_config.max_diff_steps = 20000
    forward_config.end_diff_time = 1
    forward_config.std_max = 90
    forward_config.std_min = 0.01
    forward_config.dim1 = 254
    forward_config.dim2 = 255
    forward_config.dataSize = 10000
    forward_config.isUnitInterval = False
    forward_config.isfBm = True
    forward_config.gif_save_freq = 50

    # Run experiment
    run(forward_config)
