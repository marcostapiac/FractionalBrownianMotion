import os
from typing import Tuple, Union

import numpy as np
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassDiffTrainer import DiffusionModelTrainer
from src.classes.ClassPredictor import AncestralSamplingPredictor, EulerMaruyamaPredictor
from src.classes.ClassSDESampler import SDESampler
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching


def ddp_setup(backend: str) -> None:
    """
    DDP setup to allow processes to discover and communicate with each other with TorchRun
    :param backend: Gloo vs NCCL for CPU vs GPU, respectively
    :return: None
    """
    init_process_group(backend=backend)


def prepare_data(data: np.ndarray, batch_size: int, config: ConfigDict) -> DataLoader:
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
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 sampler=DistributedSampler(train))
    else:
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=True,
                                 num_workers=0)

    return trainLoader


@record
def train_and_save_diffusion_model(data: np.ndarray,
                                   config: ConfigDict,
                                   diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                                   scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching]) -> None:
    """
    Helper function to initiate training
        :param rank: Unique process indentifier
        :param world_size: Total number of processes
        :param data: Dataset
        :param config: Configuration dictionary with relevant parameters
        :param diffusion: SDE model
        :param scoreModel: Score network architecture
        :return: None
    """
    if config.has_cuda:
        ddp_setup(backend="nccl")
        device = int(os.environ["LOCAL_RANK"])
    else:
        ddp_setup(backend="gloo")
        device = torch.device("cpu")

    # Preprocess data
    trainLoader = prepare_data(data=data, batch_size=config.batch_size, config=config)

    # Define optimiser
    optimiser = torch.optim.Adam((scoreModel.parameters()), lr=config.lr)  # TODO: Do we not need DDP?

    # Define trainer
    train_eps, end_diff_time, max_diff_steps, checkpoint_freq = config.train_eps, config.end_diff_time, config.max_diff_steps, config.save_freq

    # TODO: When using DDP, set device = rank passed by mp.spawn OR by torchrun
    trainer = DiffusionModelTrainer(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                    checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                    loss_aggregator=torchmetrics.aggregation.MeanMetric,
                                    snapshot_path=config.scoreNet_snapshot_path, device=device,
                                    train_eps=train_eps,
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps)

    # Start training
    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)

    # Cleanly exit the DDP training
    destroy_process_group()


@record
def reverse_sampling(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                     scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], data_shape: Tuple[int, int],
                     config: ConfigDict) -> torch.Tensor:
    """
    Helper function to initiate sampling
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    # TODO: DDP cannot be used here since sampling is sequential, so only single-machine, single-GPU/CPU?
    if config.has_cuda:
        device = 0
    else:
        device = torch.device("cpu")

    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device]
    predictor = AncestralSamplingPredictor(
        *predictor_params) if config.predictor_model == "ancestral" else EulerMaruyamaPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, torch.Tensor([config.snr]), device, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = SDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor, corrector=corrector)

    # Sample
    final_samples = sampler.sample(shape=data_shape, torch_device=device)
    return final_samples  # TODO Check if need to detach