from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import reverse_sampling, evaluate_circle_performance, \
    initialise_training
from utils.math_functions import generate_circles


def run_experiment(dataSize: int, diffusion: VESDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                   config: ConfigDict) -> None:
    try:
        assert (config.train_eps <= config.sample_eps)
        circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                          data_shape=(dataSize, config.timeDim), config=config)
    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_circles(S=dataSize, noise=config.cnoise)
    evaluate_circle_performance(true_samples, circle_samples.numpy(), td=config.timeDim)

if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.circles import get_config

    config = get_config()
    td = config.timeDim
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # Training data
    trainEps = config.train_eps
    sampleEps = config.sample_eps
    N = config.max_diff_steps
    Tdiff = config.end_diff_time

    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)

    training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)

    try:
        data = np.load(config.data_path)
        assert (data.shape[0] >= training_size)
        data = data[:training_size, :]
        try:
            scoreModel.load_state_dict(torch.load(config.filename))
        except FileNotFoundError:
            initialise_training(data=data, scoreModel=scoreModel, diffusion=diffusion, config=config)


    except (AssertionError, FileNotFoundError) as e:
        data = generate_circles(S=training_size, noise=config.cnoise)
        np.save(config.data_path, data)  # TODO is this the most efficient way
        initialise_training(data=data, scoreModel=scoreModel, diffusion=diffusion, config=config)

    s = 30000
    scoreModel.load_state_dict(torch.load(config.filename))
    run_experiment(diffusion=diffusion, scoreModel=scoreModel, dataSize=s, config=config)
