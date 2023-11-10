import ml_collections
import numpy as np
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from utils.experiment_evaluations import run_fBm_perfect_VESDE_score
from utils.plotting_functions import make_gif


def run(perfect_config: ConfigDict) -> None:
    try:
        assert(perfect_config.save_freq <= perfect_config.max_diff_steps)
        assert(perfect_config.save_freq > 0)
        assert(perfect_config.dim1 < perfect_config.timeDim)
        assert(perfect_config.dim2 < perfect_config.timeDim)
    except AssertionError as e:
        raise AssertionError("Error {}; check experiment parameters\n".format(e))
    rng = np.random.default_rng()
    diffusion = VESDEDiffusion(stdMax=perfect_config.std_max, stdMin=perfect_config.std_min)
    dim_pair = torch.Tensor([perfect_config.dim1, perfect_config.dim2]).to(torch.int32)
    folder_path = project_config.ROOT_DIR + "experiments/results/backward_gifs/"

    gif_path = "perfect_fBm_dimPair{}_dimPair{}_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_StdMax{:.4e}_StdMin{:.4e}".format(dim_pair[0],
                                                                                                          dim_pair[1],
                                                                                                          perfect_config.hurst,
                                                                                                          perfect_config.timeDim,
                                                                                                          perfect_config.max_diff_steps,
                                                                                                          perfect_config.end_diff_time,
                                                                                                          perfect_config.std_max,
                                                                                                          perfect_config.std_min).replace(
        ".", "")
    run_fBm_perfect_VESDE_score(dataSize=perfect_config.dataSize, dim_pair=dim_pair, diffusion=diffusion, rng=rng,
                                perfect_config=perfect_config, folderPath=folder_path, gifPath=gif_path)

    make_gif(folder_path, gif_path)


if __name__ == "__main__":
    # Data parameters
    config = ml_collections.ConfigDict()
    config.has_cuda = torch.cuda.is_available()
    config.predictor_model = "ancestral"
    config.hurst = 0.7
    config.timeDim = 256
    config.max_diff_steps = 1000
    config.end_diff_time = 1
    config.std_max = 5.
    config.std_min = 0.01
    config.dim1 = 254
    config.dim2 = 255
    config.dataSize = 5000
    config.save_freq = 10

    # Run experiments
    run(config)
