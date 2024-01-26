import os
import pickle
from typing import Union

import numpy as np
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, DistributedSampler

from src.classes.ClassConditionalDiffTrainer import ConditionalDiffusionModelTrainer
from src.generative_modelling.data_processing import prepare_scoreModel_data
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import run_fBm_experiment
from utils.math_functions import generate_fBn

def prepare_recursive_scoreModel_data(data: np.ndarray, batch_size: int, config: ConfigDict) -> DataLoader:
    """
    Split data into train, eval, test sets and create DataLoaders for training
        :param data: Training data
        :param batch_size: Batch size
        :param config: ML Collection dictionary
        :return: Train, Validation, Test dataloaders
    """
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    train, _, _ = torch.utils.data.random_split(dataset, [1., 0., 0.])
    if config.has_cuda:
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=False, shuffle=False,
                                 sampler=DistributedSampler(train))
    else:
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=True,
                                 num_workers=0)
    print("Total Number of Datapoints {} :: DataLoader Total Number of Datapoints {}".format(data.shape[0], len(trainLoader.sampler)))
    return trainLoader

@record
def train_and_save_recursive_diffusion_model(data: np.ndarray,
                                   config: ConfigDict,
                                   diffusion: VPSDEDiffusion,
                                   scoreModel: Union[NaiveMLP, ConditionalTimeSeriesScoreMatching]) -> None:
    """
    Helper function to initiate training
        :param data: Dataset
        :param config: Configuration dictionary with relevant parameters
        :param diffusion: SDE model
        :param scoreModel: Score network architecture
        :return: None
    """
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        device = torch.device("cpu")

    # Preprocess data
    trainLoader = prepare_recursive_scoreModel_data(data=data, batch_size=config.batch_size, config=config)
    # Define optimiser
    optimiser = torch.optim.Adam((scoreModel.parameters()), lr=config.lr)  # TODO: Do we not need DDP?

    # Define trainer
    train_eps, end_diff_time, max_diff_steps, checkpoint_freq = config.train_eps, config.end_diff_time, config.max_diff_steps, config.save_freq

    # TODO: When using DDP, set device = rank passed by mp.spawn OR by torchrun
    trainer = ConditionalDiffusionModelTrainer(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                    checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                    loss_aggregator=torchmetrics.aggregation.MeanMetric,
                                    snapshot_path=config.scoreNet_snapshot_path, device=device,
                                    train_eps=train_eps,
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps, to_weight=config.weightings, hybrid_training=config.hybrid)

    # Start training
    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)


if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07 import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)

    rng = np.random.default_rng()
    scoreModel = ConditionalTimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    #init_experiment(config=config)

    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(min(1 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000))
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fBn(T=config.timeDim, isUnitInterval=config.isUnitInterval, S=training_size, H=config.hurst)
            np.save(config.data_path, data)
        if config.isfBm:
            data = data.cumsum(axis=1)[:training_size, :]
        else:
            data = data[:training_size, :]
        data = np.atleast_3d(data)
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        train_and_save_recursive_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))

    cleanup_experiment()

    run_fBm_experiment(dataSize=config.dataSize, diffusion=diffusion, scoreModel=scoreModel, rng=rng, config=config)
