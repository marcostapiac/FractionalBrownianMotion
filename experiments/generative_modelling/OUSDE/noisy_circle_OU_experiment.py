from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import reverse_sampling, evaluate_circle_performance, \
    initialise_training
from utils.math_functions import generate_circles


def run_experiment(dataSize: int, diffusion: OUSDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                   config: ConfigDict) -> None:
    try:
        assert (config.train_eps <= config.sample_eps)
        circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                          data_shape=(dataSize, config.timeDim), config=config)

    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_circles(S=training_size, noise=config.cnoise)
    evaluate_circle_performance(true_samples, circle_samples.cpu().numpy(), config=config)


if __name__ == "__main__":
    # Data parameters
    from configs.OU.circles import get_config

    config = get_config()
    td = config.timeDim

    # Training data
    trainEps = config.train_eps
    sampleEps = config.sample_eps
    N = config.max_diff_steps
    Tdiff = config.end_diff_time

    modelFileName = config.filename
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = OUSDEDiffusion()

    training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)

    try:
        data = np.load(config.data_path)
        assert (data.shape[0] >= training_size)
        try:
            scoreModel.load_state_dict(torch.load(modelFileName))
        except FileNotFoundError:
            initialise_training(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)

    except (AssertionError, FileNotFoundError) as e:
        print("Generating synthetic data\n")
        data = generate_circles(S=training_size, noise=config.cnoise)
        np.save(config.data_path, data)  # TODO is this the most efficient way
        initialise_training(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)

    s = 3000
    scoreModel.load_state_dict(torch.load(modelFileName))
    run_experiment(diffusion=diffusion, scoreModel=scoreModel, dataSize=s, config=config)
