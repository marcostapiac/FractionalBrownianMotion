from typing import Union

import numpy as np
import torch
from ml_collections import ConfigDict

from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import revamped_train_and_save_diffusion_model, reverse_sampling, evaluate_circle_performance
from utils.math_functions import generate_circles


def run_experiment(dataSize: int, diffusion: VPSDEDiffusion, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
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
    from configs.VPSDE.circles import get_config

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
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)

    try:
        data = np.load(config.data_path)
        assert (data.shape[0] >= training_size)
        data = data[:training_size, :]
        try:
            scoreModel.load_state_dict(torch.load(modelFileName))
        except FileNotFoundError:
            scoreModel = revamped_train_and_save_diffusion_model(data, model_filename=modelFileName,
                                                                 batch_size=config.batch_size,
                                                                 nEpochs=config.max_epochs, lr=config.lr,
                                                                 train_eps=trainEps,
                                                                 diffusion=diffusion, scoreModel=scoreModel,
                                                                 checkpoint_freq=config.save_freq, max_diff_steps=N,
                                                                 end_diff_time=Tdiff)

    except (AssertionError, FileNotFoundError) as e:
        data = generate_circles(S=training_size, noise=config.cnoise)
        np.save(config.data_path, data)  # TODO is this the most efficient way
        scoreModel = revamped_train_and_save_diffusion_model(data, model_filename=modelFileName,
                                                             batch_size=config.batch_size, nEpochs=config.max_epochs,
                                                             lr=config.lr, train_eps=trainEps,
                                                             diffusion=diffusion, scoreModel=scoreModel,
                                                             checkpoint_freq=config.save_freq, max_diff_steps=N,
                                                             end_diff_time=Tdiff)
    s = 30000
    run_experiment(diffusion=diffusion, scoreModel=scoreModel, dataSize=s, config=config)
