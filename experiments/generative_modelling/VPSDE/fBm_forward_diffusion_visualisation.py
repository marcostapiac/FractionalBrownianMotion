import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from utils.math_functions import compute_fBm_cov, compute_fBn_cov
from utils.plotting_functions import make_gif, plot_and_save_diffused_fBm_snapshot


def run(perfect_config: ConfigDict) -> None:
    try:
        assert (perfect_config.gif_save_freq <= perfect_config.max_diff_steps)
        assert (perfect_config.gif_save_freq > 0)
        assert (perfect_config.dim1 < perfect_config.ts_length)
        assert (perfect_config.dim2 < perfect_config.ts_length)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    diffusion = VPSDEDiffusion(beta_max=perfect_config.beta_max, beta_min=perfect_config.beta_min)
    gen = FractionalBrownianNoise(perfect_config.hurst, np.random.default_rng())

    dim_pair = torch.Tensor([forward_config.dim1, forward_config.dim2]).to(torch.int32)

    gen = FractionalBrownianNoise(forward_config.hurst, np.random.default_rng())
    if forward_config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(gen, T=forward_config.ts_length, isUnitInterval=forward_config.isUnitInterval)).to(
            torch.float32)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(gen, T=forward_config.ts_length, isUnitInterval=forward_config.isUnitInterval)).to(
            torch.float32)

    org_data = []
    for _ in tqdm(range(forward_config.dataSize)):
        tmp = gen.circulant_simulation(forward_config.ts_length, scaleUnitInterval=forward_config.isUnitInterval)
        if forward_config.isfBm: tmp = tmp.cumsum()
        org_data.append(tmp)

    org_data = torch.from_numpy(np.array(org_data)).to(torch.float32)

    # Now choose the dimensions we are interested in
    org_data = torch.index_select(org_data, dim=1, index=dim_pair)
    data_cov = torch.index_select(torch.index_select(data_cov, dim=0, index=dim_pair), dim=1, index=dim_pair)

    ts = np.linspace(0., perfect_config.end_diff_time, num=perfect_config.max_diff_steps)
    folder_path = project_config.ROOT_DIR + "experiments/results/forward_gifs/"
    gif_path = "{}_incs_fBm_dimPair{}_dimPair{}_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}".format(
        not forward_config.isfBm, forward_config.isUnitInterval, dim_pair[0],
        dim_pair[1],
        perfect_config.hurst,
        perfect_config.ts_length,
        perfect_config.max_diff_steps,
        perfect_config.end_diff_time,
        perfect_config.beta_max,
        perfect_config.beta_min).replace(
        ".", "")

    for i in tqdm(range(perfect_config.max_diff_steps)):
        eff_time = diffusion.get_eff_times(diff_times=torch.Tensor([ts[i]]))
        xts, _ = diffusion.noising_process(org_data, eff_time)
        if i % perfect_config.gif_save_freq == 0 or i == (perfect_config.max_diff_steps - 1):
            save_path = folder_path + gif_path + "_diffIndex_{}.png".format(i + 1)
            plot_title = "Forward VPSDE Samples with $T={}$ at time {}".format(perfect_config.ts_length,
                                                                               round((
                                                                                             i + 1) / perfect_config.max_diff_steps,
                                                                                     5))
            xlabel = "LSTM_fBm Dimension {}".format(dim_pair[0] + 1)
            ylabel = "LSTM_fBm Dimension {}".format(dim_pair[1] + 1)
            cov = (1. - torch.exp(-eff_time)) * torch.eye(2) + torch.exp(-eff_time) * data_cov
            plot_and_save_diffused_fBm_snapshot(samples=xts, cov=cov, save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    forward_config = ml_collections.ConfigDict()
    forward_config.has_cuda = torch.cuda.is_available()
    forward_config.hurst = 0.7
    forward_config.ts_length = 256
    forward_config.max_diff_steps = 10000
    forward_config.end_diff_time = 1
    forward_config.beta_max = 20
    forward_config.beta_min = 0.0001
    forward_config.dim1 = 254
    forward_config.dim2 = 255
    forward_config.dataSize = 10000
    forward_config.gif_save_freq = 10
    forward_config.isUnitInterval = False
    forward_config.isfBm = True

    # Run experiment
    run(forward_config)
