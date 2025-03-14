from typing import Union

import numpy as np
from ml_collections import ConfigDict

from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import evaluate_fBm_performance
from utils.experiment_evaluations import prepare_fBm_experiment, run_fBm_experiment
from utils.math_functions import generate_fBm


def run_experiment(dataSize: int, diffusion: OUSDEDiffusion, scoreModel: Union[NaiveMLP, TSScoreMatching],
                   rng: np.random.Generator, config: ConfigDict, experiment_res: dict) -> dict:
    try:
        assert (config.train_eps <= config.sample_eps)
        fBm_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                       data_shape=(dataSize, config.ts_length),
                                       config=config)
    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_fBm(H=config.hurst, T=config.ts_length, S=dataSize, rng=rng,
                                isUnitInterval=config.isUnitInterval)
    return evaluate_fBm_performance(true_samples, fBm_samples.cpu().numpy(), rng=rng, config=config,
                                    exp_dict=experiment_res)


if __name__ == "__main__":
    # Data parameters
    from configs.OU.fBm_T2_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)

    rng = np.random.default_rng()
    scoreModel = TSScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = OUSDEDiffusion()

    init_experiment(config=config)

    scoreModel = prepare_fBm_experiment(diffusion=diffusion, scoreModel=scoreModel, config=config, rng=rng)
    cleanup_experiment()

    run_fBm_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, rng=rng, config=config)
