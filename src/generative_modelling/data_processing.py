import os
import time
from typing import Tuple, Union
import signatory
import numpy as np
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MeanMetric

from src.classes.ClassConditionalLSTMDiffTrainer import ConditionalLSTMDiffusionModelTrainer
from src.classes.ClassConditionalMarkovianDiffTrainer import ConditionalMarkovianDiffusionModelTrainer
from src.classes.ClassConditionalSDESampler import ConditionalSDESampler
from src.classes.ClassConditionalSignatureDiffTrainer import ConditionalSignatureDiffusionModelTrainer
from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassDiffTrainer import DiffusionModelTrainer
from src.classes.ClassPredictor import AncestralSamplingPredictor, EulerMaruyamaPredictor, \
    ConditionalAncestralSamplingPredictor
from src.classes.ClassSDESampler import SDESampler
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTimeSeriesScoreMatching import \
    ConditionalMarkovianTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalSignatureTimeSeriesScoreMatching import \
    ConditionalSignatureTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.math_functions import compute_sig_size, ts_signature_pipeline, tensor_algebra_product, time_aug


def prepare_scoreModel_data(data: np.ndarray, batch_size: int, config: ConfigDict) -> DataLoader:
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
    print("Total Number of Datapoints {} :: DataLoader Total Number of Datapoints {}".format(data.shape[0],
                                                                                             len(trainLoader.sampler)))
    return trainLoader


@record
def train_and_save_diffusion_model(data: np.ndarray,
                                   config: ConfigDict,
                                   diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                                   scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching]) -> None:
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
    trainLoader = prepare_scoreModel_data(data=data, batch_size=config.batch_size, config=config)

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
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                    to_weight=config.weightings, hybrid_training=config.hybrid)

    # Start training
    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)


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
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
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
    final_samples = sampler.sample(shape=data_shape, torch_device=device, early_stop_idx=config.early_stop_idx)
    return final_samples  # TODO Check if need to detach


def compute_current_sig_feature(ts_time:int, device:Union[int, str],past_feat:torch.Tensor, basepoint:torch.Tensor,latest_path:torch.Tensor, config:ConfigDict, score_network:ConditionalSignatureTimeSeriesScoreMatching)->torch.Tensor:
    """
    Efficient computation of path signature through concatenation
        :param ts_time: Current time series time
        :param device: Device on which tensors are stored
        :param past_feat: Last feature
        :param latest_path: Last path value
        :param config: ML experiment configuration file
    :return: Concactenated path feature
    """
    # ts_time: we have generated x1:x_tstime and want to generate x_(ts_time+1)
    # ts_time == 0: we have generated nothing
    # ts_time == 1: we have generated x1, want to generate x2
    T = config.ts_length
    assert(len(basepoint.shape)==len(latest_path.shape)==3)
    basepoint = torch.zeros_like(basepoint) if ts_time <= 1  else basepoint
    latest_path = torch.zeros_like(latest_path) if ts_time == 0  else latest_path
    if isinstance(past_feat.device, int):
        increment_sig = score_network.module.signet.forward(latest_path, time_ax=torch.atleast_2d(
            torch.Tensor([ts_time]) / T).T, basepoint=time_aug(basepoint, time_ax=torch.atleast_2d(
            torch.Tensor([max(ts_time - 1, 0)]) / T).T.to(device)))
    else:
        increment_sig = score_network.signet.forward(latest_path, time_ax=torch.atleast_2d(
            torch.Tensor([ts_time]) / T).T, basepoint=time_aug(basepoint, time_ax=torch.atleast_2d(
            torch.Tensor([max(ts_time - 1, 0)]) / T).T.to(device)))
    if ts_time >= 1:
        # past_feat = features[[0], [ts_time - 1], :]  # Feature for generating x1 (using x0 only)
        curr_feat = signatory.signature_combine(sigtensor1=past_feat.squeeze(dim=1),
                                                sigtensor2=increment_sig.squeeze(dim=1),
                                                input_channels=2, depth=5)
        print(curr_feat)
    else:
        curr_feat = increment_sig
    curr_feat = curr_feat.unsqueeze(dim=1)
    print(curr_feat.shape, past_feat.shape)
    assert (curr_feat.shape == past_feat.shape)
    return curr_feat




@record
def recursive_signature_reverse_sampling(diffusion: VPSDEDiffusion,
                                    scoreModel: ConditionalSignatureTimeSeriesScoreMatching, data_shape: Tuple[int, int, int],
                                    config: ConfigDict) -> torch.Tensor:
    """
    Recursive reverse sampling using path signatures
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    assert (config.predictor_model == "ancestral")
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
    predictor = ConditionalAncestralSamplingPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, torch.Tensor([config.snr]), device, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = ConditionalSDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor,
                                    corrector=corrector)
    scoreModel.eval()
    with torch.no_grad():
        paths = [torch.zeros(size=(data_shape[0],1,data_shape[-1])).to(device)] # Initial starting point (can be set to anything)
        output = torch.zeros((data_shape[0],1, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc)-1)).to(device)
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            output = compute_current_sig_feature(ts_time=t, device=device,past_feat=output, basepoint=paths[max(t - 2,0)],latest_path=paths[max(0,t - 1)], config=config, score_network=scoreModel)
            samples = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=output,
                                     early_stop_idx=config.early_stop_idx)
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)[:,1:]
    assert(final_paths.shape == (data_shape[0], data_shape[1]))
    return np.atleast_2d(final_paths.numpy())

@record
def recursive_LSTM_reverse_sampling(diffusion: VPSDEDiffusion,
                                    scoreModel: ConditionalTimeSeriesScoreMatching, data_shape: Tuple[int, int, int],
                                    config: ConfigDict) -> torch.Tensor:
    """
    Recursive reverse sampling using LSTMs
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    assert (config.predictor_model == "ancestral")
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
    predictor = ConditionalAncestralSamplingPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, torch.Tensor([config.snr]), device, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = ConditionalSDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor,
                                    corrector=corrector)

    scoreModel.eval()
    with torch.no_grad():
        samples = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
        paths = []
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            if t == 0:
                output, (h, c) = scoreModel.rnn(samples, None)
            else:
                output, (h, c) = scoreModel.rnn(samples, (h, c))
            samples = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=output,
                                     early_stop_idx=config.early_stop_idx)
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)
    return np.atleast_2d(final_paths.numpy())

@record
def recursive_markovian_reverse_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalTimeSeriesScoreMatching,
                                         data_shape: Tuple[int, int, int],
                                         config: ConfigDict) -> torch.Tensor:
    """
    Recursive reverse sampling using Markovian Diffusion Model
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return: Final reverse-time samples
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    assert (config.predictor_model == "ancestral")
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
    predictor = ConditionalAncestralSamplingPredictor(*predictor_params)

    # Define corrector
    corrector_params = [config.max_lang_steps, torch.Tensor([config.snr]), device, diffusion]
    if config.corrector_model == "VE":
        corrector = VESDECorrector(*corrector_params)
    elif config.corrector_model == "VP":
        corrector = VPSDECorrector(*corrector_params)
    else:
        corrector = None
    sampler = ConditionalSDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor,
                                    corrector=corrector)

    scoreModel.eval()
    with torch.no_grad():
        paths = []
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            if t == 0:
                features = torch.zeros(size=(data_shape[0], 1, config.mkv_blnk * config.ts_dims)).to(device)
            else:
                past = [torch.zeros_like(paths[0]) for _ in range(max(0, config.mkv_blnk - t))] + paths[
                                                                                                  -config.mkv_blnk:]
                features = torch.stack(past, dim=2).reshape(
                    (data_shape[0], 1, config.mkv_blnk * config.ts_dims, 1)).squeeze(-1)
            samples = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=features,
                                     early_stop_idx=config.early_stop_idx)
            # Samples are size (BatchSize, 1, TimeSeriesDimension)
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)
    return np.atleast_2d(final_paths.numpy())


def prepare_recursive_scoreModel_data(data: Union[np.ndarray, torch.Tensor], batch_size: int, config: ConfigDict) -> DataLoader:
    """
    Split data into train, eval, test sets and create DataLoaders for training
        :param data: Training data
        :param batch_size: Batch size
        :param config: ML Collection dictionary
        :return: Train, Validation, Test dataloaders
    """
    try:
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(data).float())
    except TypeError as e:
        dataset = torch.utils.data.TensorDataset(data)
    L = data.shape[0]
    train, _, _ = torch.utils.data.random_split(dataset, [L, 0, 0])
    if config.has_cuda:
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=False, shuffle=False,
                                 sampler=DistributedSampler(train))
    else:
        trainLoader = DataLoader(train, batch_size=batch_size, pin_memory=True, shuffle=True,
                                 num_workers=0)
    print("Total Number of Datapoints {} :: DataLoader Total Number of Datapoints {}".format(data.shape[0],
                                                                                             len(trainLoader.sampler)))
    return trainLoader


@record
def train_and_save_recursive_diffusion_model(data: np.ndarray,
                                             config: ConfigDict,
                                             diffusion: VPSDEDiffusion,
                                             scoreModel: Union[
                                                 NaiveMLP, ConditionalTimeSeriesScoreMatching, ConditionalTimeSeriesScoreMatching, ConditionalMarkovianTimeSeriesScoreMatching],
                                             trainClass: Union[
                                                 ConditionalLSTMDiffusionModelTrainer, ConditionalMarkovianDiffusionModelTrainer, ConditionalSignatureDiffusionModelTrainer, DiffusionModelTrainer]) -> None:
    """
    Helper function to initiate training for recursive diffusion model
        :param data: Dataset
        :param config: Configuration dictionary with relevant parameters
        :param diffusion: SDE model
        :param scoreModel: Score network architecture
        :param trainClass: Class of diffusion trainer
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
    try:
        # Markovian
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps, to_weight=config.weightings,
                             mkv_blnk=config.mkv_blnk,
                             hybrid_training=config.hybrid)
        # Start training
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
    except (AttributeError, KeyError) as e:
        # Signature
        print(e)
        try:
            trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                 checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                 loss_aggregator=MeanMetric,
                                 snapshot_path=config.scoreNet_snapshot_path, device=device,
                                 train_eps=train_eps,
                                 end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                 to_weight=config.weightings,
                                 hybrid_training=config.hybrid)
            # Start training
            trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path, ts_dims=config.ts_dims)
        except (AttributeError, KeyError) as e:
            # LSTM
            trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                 checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                 loss_aggregator=MeanMetric,
                                 snapshot_path=config.scoreNet_snapshot_path, device=device,
                                 train_eps=train_eps,
                                 end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                 to_weight=config.weightings,
                                 hybrid_training=config.hybrid)

            # Start training
            trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)