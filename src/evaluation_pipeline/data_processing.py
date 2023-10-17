import os
from typing import Tuple

import numpy as np
import sklearn.metrics
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed import destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTM import DiscriminativeLSTM
from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTMDataset import DiscriminativeLSTMDataset
from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTMInference import \
    DiscriminativeLSTMInference
from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTMTrainer import DiscriminativeLSTMTrainer
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTM import PredictiveLSTM
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTMDataset import PredictiveLSTMDataset
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTMInference import PredictiveLSTMInference
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTMTrainer import PredictiveLSTMTrainer
from src.generative_modelling.data_processing import ddp_setup


def prepare_predLSTM_data(data: np.ndarray, config: ConfigDict) -> Tuple[Dataset, DataLoader]:
    """
    Prepare data loaders to feed into predictive LSTM
        :param data: Synthetic time-series
        :return:
            1. New dataset
            2. Corresponding dataloader
    """
    dataset = PredictiveLSTMDataset(data=data, lookback=config.lookback)
    if config.has_cuda:
        loader = DataLoader(dataset, batch_size=config.lstm_batch_size, pin_memory=True, shuffle=False,
                            sampler=DistributedSampler(dataset))
    else:
        loader = DataLoader(dataset, batch_size=config.lstm_batch_size, pin_memory=True, shuffle=True,
                            num_workers=0)

    return dataset, loader


@record
def train_and_save_predLSTM(data: np.ndarray, config: ConfigDict, model: PredictiveLSTM) -> None:
    """
    Save an LSTM model trained to predict 1-step ahead values
        :param data: N x T time-series matrix
        :param config: ML experiment configuration file
        :param model: Untrained model
        :return: None
    """
    if config.has_cuda:
        ddp_setup(backend="nccl")
        device = int(os.environ["LOCAL_RANK"])
    else:
        ddp_setup(backend="gloo")
        device = torch.device("cpu")
    _, trainLoader = prepare_predLSTM_data(data=data, config=config)

    assert (list(next(iter(trainLoader))[0].shape[1:]) == [config.lookback, 1])

    # Define optimiser
    optimiser = torch.optim.Adam((model.parameters()), lr=config.lr)  # TODO: Do we not need DDP?

    # Define trainer
    # TODO: When using DDP, set device = rank passed by mp.spawn OR by torchrun
    trainer = PredictiveLSTMTrainer(model=model, train_data_loader=trainLoader,
                                    checkpoint_freq=config.save_freq, optimiser=optimiser, loss_fn=torch.nn.L1Loss,
                                    loss_aggregator=torchmetrics.aggregation.MeanMetric,
                                    snapshot_path=config.pred_lstm_snapshot_path, device=device)

    # Start training
    trainer.train(max_epochs=config.lstm_max_epochs, model_filename=config.pred_lstm_trained_path)

    # Cleanly exit the DDP training
    destroy_process_group()


def test_predLSTM(original_data: np.ndarray, synthetic_data: np.ndarray, config: ConfigDict,
                  model: PredictiveLSTM) -> Tuple[float, float]:
    """
    Test trained predictive LSTM on both true samples and synthetic samples
        :param original_data: Exact samples from desired distribution
        :param synthetic_data: Synthetic samples from reverse-diffusion
        :param config: ML condfiguration file
        :param model: Empty model
        :return: MAE losses for original and synthetic datasets
    """
    try:
        model.load_state_dict(torch.load(config.pred_lstm_trained_path))
    except FileNotFoundError as e:
        print("Please train predictive LSTM before testing\n")
    finally:
        if config.has_cuda:
            ddp_setup(backend="nccl")
            device = int(os.environ["LOCAL_RANK"])
        else:
            ddp_setup(backend="gloo")
            device = torch.device("cpu")

        # Instantiate sampler
        inference = PredictiveLSTMInference(model=model, device=device,
                                            loss_fn=torchmetrics.SymmetricMeanAbsolutePercentageError,
                                            loss_aggregator=torchmetrics.MeanMetric)

        # Prepare data
        org_dataset, org_loader = prepare_predLSTM_data(original_data, config)
        synth_dataset, synth_loader = prepare_predLSTM_data(synthetic_data, config)

        # Run forward model
        print("Running with original samples\n")
        org_loss = inference.run(org_loader)
        print("Running with synthetic samples\n")
        synth_loss = inference.run(synth_loader)

        print("Average SMAPE :: Original vs Synthetic :: {} vs {}".format(round(org_loss, 5), round(synth_loss, 5)))
        return round(org_loss, 5), round(synth_loss, 5)


def prepare_discLSTM_data(original: np.ndarray, synthetic: np.ndarray, labels: list, config: ConfigDict) -> Tuple[
    Dataset, DataLoader]:
    """
    Prepare data loaders to feed into discriminative LSTM
        :param originals: Original / exact time-series
        :param synthetic: Synthetic time-series
        :return:
            1. New dataset
            2. Corresponding dataloader
    """
    dataset = DiscriminativeLSTMDataset(org_data=original, synth_data=synthetic, labels=labels)
    if config.has_cuda:
        loader = DataLoader(dataset, batch_size=config.lstm_batch_size, pin_memory=True, shuffle=False,
                            sampler=DistributedSampler(dataset))
    else:
        loader = DataLoader(dataset, batch_size=config.lstm_batch_size, pin_memory=True, shuffle=True,
                            num_workers=0)

    return dataset, loader


@record
def train_and_save_discLSTM(org_data: np.ndarray, synth_data: np.ndarray, config: ConfigDict,
                            model: DiscriminativeLSTM) -> None:
    """
    Save a trained discriminative LSTM
        :param org_data: Exact samples from distribution
        :param synth_data: Generated samples
        :param config: ML experiment configuration file
        :param model: Untrained model
        :return: None
    """
    if config.has_cuda:
        ddp_setup(backend="nccl")
        device = int(os.environ["LOCAL_RANK"])
    else:
        ddp_setup(backend="gloo")
        device = torch.device("cpu")

    _, trainLoader = prepare_discLSTM_data(original=org_data, synthetic=synth_data, config=config, labels=[1, 0])

    # Define optimiser
    optimiser = torch.optim.Adam((model.parameters()), lr=config.lr)  # TODO: Do we not need DDP?

    # Define trainer
    # TODO: When using DDP, set device = rank passed by mp.spawn OR by torchrun
    trainer = DiscriminativeLSTMTrainer(model=model, train_data_loader=trainLoader,
                                        checkpoint_freq=config.save_freq, optimiser=optimiser,
                                        loss_fn=torch.nn.BCEWithLogitsLoss,
                                        loss_aggregator=torchmetrics.aggregation.MeanMetric,
                                        snapshot_path=config.disc_lstm_snapshot_path, device=device)

    # Start training
    trainer.train(max_epochs=config.lstm_max_epochs, model_filename=config.disc_lstm_trained_path)

    # Cleanly exit the DDP training
    destroy_process_group()


def test_discLSTM(original_data: np.ndarray, synthetic_data: np.ndarray, config: ConfigDict,
                  model: DiscriminativeLSTM) -> Tuple[float, float]:
    """
    Test trained predictive LSTM on both true samples and synthetic samples
        :param org_data: Exact samples from desired distribution
        :param synth_data: Synthetic samples from reverse-diffusion
        :param config: ML condfiguration file
        :param model: Empty discriminative model
        :return: Original and synthetic dataset losses
    """
    try:
        model.load_state_dict(torch.load(config.disc_lstm_trained_path))
    except FileNotFoundError as e:
        print("Please train discriminative LSTM before testing\n")
    finally:
        if config.has_cuda:
            ddp_setup(backend="nccl")
            device = int(os.environ["LOCAL_RANK"])
        else:
            ddp_setup(backend="gloo")
            device = torch.device("cpu")

        # Instantiate sampler
        inference = DiscriminativeLSTMInference(model=model, device=device, loss_fn=sklearn.metrics.accuracy_score,
                                                loss_aggregator=torchmetrics.MeanMetric)

        # Prepare data
        L = original_data.shape[0]
        org_dataset, org_loader = prepare_discLSTM_data(original_data[:L // 2], original_data[L // 2:], config=config,
                                                        labels=[1, 1])
        synth_dataset, synth_loader = prepare_discLSTM_data(synthetic_data[:L // 2], synthetic_data[L // 2:],
                                                            config=config, labels=[0, 0])

        # Run forward model
        print("Running with original samples\n")
        org_acc = inference.run(org_loader)
        print("Running with synthetic samples\n")
        synth_loss = inference.run(synth_loader)  # Loss because we provide labels of 0 but test if they are actually 1

        print("Average Success vs Error Rate :: Original vs Synthetic :: {} vs {}".format(round(org_acc, 5),
                                                                                          round(synth_loss, 5)))
        return round(org_acc, 5), round(synth_loss, 5)
