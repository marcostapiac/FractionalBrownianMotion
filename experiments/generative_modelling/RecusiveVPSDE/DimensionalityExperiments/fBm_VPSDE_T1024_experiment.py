import numpy as np

from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import run_fBm_experiment, prepare_fBm_experiment

if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.fBm_T1024_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)

    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)

    scoreModel = prepare_fBm_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config, rng=rng)

    cleanup_experiment()

    run_fBm_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, rng=rng, config=config)
