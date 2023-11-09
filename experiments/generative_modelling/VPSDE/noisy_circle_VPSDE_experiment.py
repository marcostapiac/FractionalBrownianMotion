from typing import Union

from ml_collections import ConfigDict

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import generate_circles
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import evaluate_circle_performance
from utils.experiment_evaluations import prepare_circle_experiment, run_circle_experiment


def run_experiment(dataSize: int, diffusion: VPSDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                   config: ConfigDict, experiment_res: dict) -> dict:
    try:
        assert (config.train_eps <= config.sample_eps)
        true_samples = generate_circles(S=dataSize, noise=config.cnoise)
        circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                          data_shape=(dataSize, config.timeDim), config=config)
    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    return evaluate_circle_performance(true_samples, circle_samples.cpu().numpy(), config=config,
                                       exp_dict=experiment_res)


if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.circles import get_config

    config = get_config()

    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    init_experiment(config=config)

    scoreModel = prepare_circle_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config)

    run_circle_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, config=config)

    cleanup_experiment()
