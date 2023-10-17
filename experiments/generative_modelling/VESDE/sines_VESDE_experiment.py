import numpy as np

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import prepare_sines_experiment, \
    run_sines_experiment

if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.sines import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)

    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)

    init_experiment(config=config)

    scoreModel = prepare_sines_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config, rng=rng)

    run_sines_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, rng=rng, config=config)

    cleanup_experiment()