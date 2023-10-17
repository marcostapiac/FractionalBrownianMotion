import numpy as np

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import prepare_circle_experiment, run_circle_experiment

if __name__ == "__main__":
    # Data parameters
    from configs.OU.circles import get_config

    config = get_config()

    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = OUSDEDiffusion()

    scoreModel = prepare_circle_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config)

    run_circle_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, config=config)
