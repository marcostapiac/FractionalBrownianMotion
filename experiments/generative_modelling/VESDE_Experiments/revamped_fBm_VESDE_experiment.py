import pickle
from typing import Union

import numpy as np
import torch

from src.generative_modelling.models import ClassVESDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils import project_config
from utils.data_processing import save_and_train_diffusion_model, evaluate_performance, \
    revamped_train_and_save_diffusion_model, reverse_sampling
from utils.math_functions import generate_fBn, generate_fBm


def run_experiment(hurst: float, timeDim: int, dataSize: int, diffusion: VESDEDiffusion, scoreModel:Union[NaiveMLP, TimeSeriesScoreMatching],
                   rng: np.random.Generator, trainEps:float, sampleEps: float) -> None:
    assert (0. < hurst < 1.)
    try:
            assert(trainEps <= sampleEps)
            fBm_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel, end_diff_time=Tdiff, max_diff_steps=N, N_lang=0, snr=0.00001, sampleEps=sampleEps, data_shape=(dataSize,timeDim))
    except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")

    true_samples = generate_fBm(H=hurst, T=timeDim, S=dataSize, rng=rng)
    evaluate_performance(true_samples, fBm_samples.numpy(), h=hurst, td=timeDim, rng=rng, unitInterval=True, annot=True,
                         evalMarginals=True, isfBm=True, permute_test=False)


if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.fBm_T2_H07 import get_config

    config = get_config()
    h = config.hurst
    td = config.timeDim

    # Training data
    trainEps = config.train_eps
    sampleEps = config.sample_eps
    N = config.max_diff_steps
    Tdiff = config.end_diff_time

    # Model parameters
    std_max = config.std_max
    std_min = config.std_min

    modelFileName = config.mlpFileName if config.model_choice == "MLP" else config.tsmFileName
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice =="TSM" else NaiveMLP(*config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=std_max, stdMin=std_min)

    training_size = min(10*sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)

    try:
        data = np.load(config.data_path)
        assert(data.shape[0] >= training_size)
        data = data[:training_size,:].cumsum(axis=1)
        try:
            scoreModel.load_state_dict(torch.load(modelFileName))
        except FileNotFoundError:
            scoreModel = revamped_train_and_save_diffusion_model(data, model_filename=modelFileName,
                                                    batch_size=config.batch_size, nEpochs=config.max_epochs, lr=config.lr, train_eps=trainEps,
                                                    diffusion=diffusion, scoreModel=scoreModel, checkpoint_freq=config.save_freq, max_diff_steps=N, end_diff_time=Tdiff)

    except (AssertionError, FileNotFoundError) as e:
        data = generate_fBn(T=td, S=training_size, H=h, rng=rng)
        np.save(config.data_path, data) # TODO is this the most efficient way
        data = data.cumsum(axis=1)
        scoreModel = revamped_train_and_save_diffusion_model(data, model_filename=modelFileName,
                                                    batch_size=config.batch_size, nEpochs=config.max_epochs, lr=config.lr, train_eps=trainEps,
                                                    diffusion=diffusion, scoreModel=scoreModel, checkpoint_freq=config.save_freq, max_diff_steps=N, end_diff_time=Tdiff)
    s = 10000
    data = data[:s, :]
    run_experiment(diffusion=diffusion, scoreModel = scoreModel, timeDim=td, dataSize=s,trainEps=trainEps,sampleEps=sampleEps, hurst=h, rng=rng)
