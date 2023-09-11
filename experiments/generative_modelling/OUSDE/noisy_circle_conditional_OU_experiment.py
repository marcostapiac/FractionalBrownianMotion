from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import revamped_train_and_save_diffusion_model, \
    evaluate_circle_performance, reverse_sampling
from utils.math_functions import generate_circles

LR = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 64


def observation_model(latent: np.ndarray) -> np.ndarray:
    return latent + 6. * np.ones_like(latent) + np.sqrt(0.1) * np.random.randn(*latent.shape)


def run_experiment(dataSize: int, diffusion: OUSDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                   config: ConfigDict) -> None:
    """ Perform ancestral sampling given model """
    try:
        assert (config.train_eps <= config.sample_eps)
        circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                          data_shape=(dataSize, config.timeDim), config=config)
    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_circles(S=dataSize, noise=config.cnoise)
    evaluate_circle_performance(true_samples, circle_samples.numpy(), td=config.timeDim)


if __name__ == "__main__":
    from configs.OU.circles_conditional import get_config

    config = get_config()
    td = config.timeDim

    # Training data
    trainEps = config.train_eps
    sampleEps = config.sample_eps
    N = config.max_diff_steps
    Tdiff = config.end_diff_time

    modelFileName = config.mlpFileName if config.model_choice == "MLP" else config.tsmFileName
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = OUSDEDiffusion()

    training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)

    try:
        latent = np.load(config.data_path)
        assert (latent.shape[0] >= training_size)
        observations = torch.from_numpy(observation_model(latent)).to(torch.float32)
        try:
            scoreModel.load_state_dict(torch.load(modelFileName))
        except FileNotFoundError:
            scoreModel = revamped_train_and_save_diffusion_model(latent, model_filename=modelFileName,
                                                                 batch_size=config.batch_size,
                                                                 nEpochs=config.max_epochs, lr=config.lr,
                                                                 train_eps=trainEps,
                                                                 diffusion=diffusion, scoreModel=scoreModel,
                                                                 checkpoint_freq=config.save_freq, max_diff_steps=N,
                                                                 end_diff_time=Tdiff)
    except (AssertionError, FileNotFoundError) as e:
        latent = generate_circles(S=training_size, noise=config.cnoise)
        np.save(config.data_path, latent)  # TODO is this the most efficient way
        observations = observation_model(latent)
        scoreModel = revamped_train_and_save_diffusion_model(latent, model_filename=modelFileName,
                                                             batch_size=config.batch_size, nEpochs=config.max_epochs,
                                                             lr=config.lr, train_eps=trainEps,
                                                             diffusion=diffusion, scoreModel=scoreModel,
                                                             checkpoint_freq=config.save_freq, max_diff_steps=N,
                                                             end_diff_time=Tdiff)
    s = 30000
    run_experiment(diffusion=diffusion, scoreModel=scoreModel, dataSize=s, config=config)
