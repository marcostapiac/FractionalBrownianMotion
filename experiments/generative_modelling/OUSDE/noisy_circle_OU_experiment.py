import numpy as np

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching
from utils.data_processing import cleanup_experiment, init_experiment
from utils.experiment_evaluations import prepare_circle_experiment, run_circle_experiment

if __name__ == "__main__":
    # Data parameters
    from configs.OU.circles import get_config

    config = get_config()

    rng = np.random.default_rng()
    scoreModel = TSScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = OUSDEDiffusion()

    init_experiment(config=config)

    scoreModel = prepare_circle_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config)
    cleanup_experiment()

    run_circle_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, config=config)
