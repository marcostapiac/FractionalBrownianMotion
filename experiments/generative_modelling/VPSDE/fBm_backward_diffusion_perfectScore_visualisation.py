import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from utils.experiment_evaluations import run_fBm_perfect_score
from utils.plotting_functions import make_gif


def run(perfect_config: ConfigDict) -> None:
    try:
        assert (perfect_config.save_freq <= perfect_config.max_diff_steps)
        assert (perfect_config.save_freq > 0)
        assert (perfect_config.dim1 < perfect_config.timeDim)
        assert (perfect_config.dim2 < perfect_config.timeDim)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    rng = np.random.default_rng()
    diffusion = VPSDEDiffusion(beta_max=perfect_config.beta_max, beta_min=perfect_config.beta_min)
    dim_pair = torch.Tensor([perfect_config.dim1, perfect_config.dim2]).to(torch.int32)
    folder_path = project_config.ROOT_DIR + "experiments/results/perfect_backward_gifs/"

    gif_path = "perfect_{}_incs_{}_unitIntv_fBm_dimPair{}_dimPair{}_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_BetaMax{:.4e}_BetaMin{:.4e}".format(
        not perfect_config.isfBm, perfect_config.isUnitInterval, dim_pair[0],
        dim_pair[1],
        perfect_config.hurst,
        perfect_config.timeDim,
        perfect_config.max_diff_steps,
        perfect_config.end_diff_time,
        perfect_config.beta_max,
        perfect_config.beta_min).replace(
        ".", "")
    run_fBm_perfect_score(dataSize=perfect_config.dataSize, dim_pair=dim_pair, diffusion=diffusion, rng=rng,
                          perfect_config=perfect_config, folderPath=folder_path, gifPath=gif_path)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    perfect_config = ml_collections.ConfigDict()
    perfect_config.has_cuda = torch.cuda.is_available()
    perfect_config.predictor_model = "ancestral"
    perfect_config.hurst = 0.7
    perfect_config.timeDim = 256
    perfect_config.max_diff_steps = 1000
    perfect_config.end_diff_time = 1
    perfect_config.beta_max = 20.
    perfect_config.beta_min = 0.1
    perfect_config.dim1 = 254
    perfect_config.dim2 = 255
    perfect_config.dataSize = 10000
    perfect_config.save_freq = 10

    # Run experiments
    run(perfect_config)
