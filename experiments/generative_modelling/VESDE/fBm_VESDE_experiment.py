from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import evaluate_performance, initialise_training, reverse_sampling
from utils.math_functions import generate_fBn, generate_fBm


def run_experiment(dataSize: int, diffusion: VESDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                   rng: np.random.Generator, config: ConfigDict) -> None:
    try:
        assert (config.train_eps <= config.sample_eps)
        fBm_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel, data_shape=(s, config.timeDim),
                                       config=config)

    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBm_samples.cpu().numpy(), rng=rng, config=config)


if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.fBm_T2_H07 import get_config

    config = get_config()
    h = config.hurst
    assert (0. < config.hurst < 1.)
    td = config.timeDim

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
        scoreModel.load_state_dict(torch.load(config.filename))
    except (FileNotFoundError) as e:
        print("No valid trained model found; proceeding to training\n")
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except (FileNotFoundError) as e:
            print("Generating synthetic data\n")
            data = generate_fBn(T=td, S=training_size, H=h, rng=rng)
            np.save(config.data_path, data)  # TODO is this the most efficient way
        finally:
            data = data.cumsum(axis=1)
            initialise_training(data=data, scoreModel=scoreModel, diffusion=diffusion, config=config)
            scoreModel.load_state_dict(torch.load(config.filename))

    s = 100000
    run_experiment(diffusion=diffusion, scoreModel=scoreModel, dataSize=s, rng=rng, config=config)
