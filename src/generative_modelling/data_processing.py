import os
from typing import Tuple, Union

import numpy as np
import torch
import torchmetrics
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import MeanMetric

from src.classes.ClassConditionalMarkovianPostMeanDiffTrainer import ConditionalMarkovianPostMeanDiffTrainer
from src.classes.ClassConditionalSDESampler import ConditionalSDESampler
from src.classes.ClassConditionalStbleTgtLSTMPostMeanDiffTrainer import ConditionalStbleTgtLSTMPostMeanDiffTrainer
from src.classes.ClassConditionalStbleTgtMarkovianPostMeanDiffTrainer import \
    ConditionalStbleTgtMarkovianPostMeanDiffTrainer
from src.classes.ClassConditionalStbleTgtMarkovianScoreDiffTrainer import ConditionalStbleTgtMarkovianScoreDiffTrainer
from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassPredictor import AncestralSamplingPredictor, \
    ConditionalAncestralSamplingPredictor, ConditionalReverseDiffusionSamplingPredictor, \
    ConditionalProbODESamplingPredictor
from src.classes.ClassSDESampler import SDESampler
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching

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
def reverse_sampling(diffusion: Union[VPSDEDiffusion, VESDEDiffusion, OUSDEDiffusion],
                     scoreModel: Union[ConditionalMarkovianTSPostMeanScoreMatching], data_shape: Tuple[int, int],
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
    assert (config.predictor_model == "Ancestral")
    predictor = AncestralSamplingPredictor(*predictor_params)
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


@record
def recursive_LSTM_reverse_sampling(diffusion: VPSDEDiffusion,
                                    scoreModel: ConditionalMarkovianTSPostMeanScoreMatching, data_shape: Tuple[int, int, int],
                                    config: ConfigDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recursive reverse sampling using LSTMs
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param data_shape: Desired shape of generated samples
        :param config: Configuration dictionary for experiment
        :return:
            1. Final reverse-time samples
            2. Conditional means
            3. Conditional variances
    """
    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
    if config.predictor_model == "CondAncestral":
        predictor = ConditionalAncestralSamplingPredictor(*predictor_params)
    elif config.predictor_model == "CondReverseDiffusion":
        predictor = ConditionalReverseDiffusionSamplingPredictor(*predictor_params)
    elif config.predictor_model == "CondProbODE":
        predictor = ConditionalProbODESamplingPredictor(*predictor_params)

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
    paths = []
    means = []
    vars = []
    prev_path = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
    for t in range(config.ts_length):
        print("Sampling at real time {}\n".format(t + 1))
        if t == 0:
            output, (h, c) = scoreModel.rnn(prev_path, None)
        else:
            output, (h, c) = scoreModel.rnn(prev_path, (h, c))
        samples, mean, var = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=output,
                                            early_stop_idx=config.early_stop_idx, ts_step=1. / config.ts_length,
                                            param_time=config.param_time, prev_path=prev_path)
        assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
        if t != 0:
            est_mean = mean * config.ts_length
            prev_path = prev_path.squeeze(-1)
            assert (est_mean.shape == prev_path.shape)
            print("Estimated drift {}\n".format(est_mean))
            if "fSin" in config.data_path:
                print("fSin\n")
                print("Expected drift {}\n".format(config.mean_rev * torch.sin(prev_path)))
                print(torch.mean(est_mean), torch.std(est_mean), torch.mean(prev_path), torch.std(prev_path),
                      torch.mean(config.mean_rev * torch.sin(prev_path)),
                      torch.std(config.mean_rev * torch.sin(prev_path)))
            elif "fOU" in config.data_path:
                print("fOU\n")
                print("Expected drift {}\n".format(-config.mean_rev * (prev_path)))
                print(torch.mean(est_mean), torch.std(est_mean), torch.mean(prev_path), torch.std(prev_path),
                      torch.mean(-config.mean_rev * (prev_path)),
                      torch.std(-config.mean_rev * (prev_path)))
        paths.append(samples.detach())
        means.append(mean.detach())
        vars.append(var.detach())
        prev_path = torch.concat(paths, dim=1).sum(dim=1).unsqueeze(1)
        print(prev_path.shape)
    final_paths = np.atleast_2d(torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2).numpy())
    conditional_means = np.atleast_2d(torch.concat(means, dim=1).cpu().numpy())
    conditional_vars = np.atleast_2d(torch.concat(vars, dim=1).cpu().numpy())
    assert (final_paths.shape == conditional_means.shape == conditional_vars.shape)
    return final_paths, conditional_means, conditional_vars


@record
def recursive_markovian_reverse_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalMarkovianTSPostMeanScoreMatching,
                                         data_shape: Tuple[int, int, int],
                                         config: ConfigDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # Define predictor
    predictor_params = [diffusion, scoreModel, config.end_diff_time, config.max_diff_steps, device, config.sample_eps]
    if config.predictor_model == "CondAncestral":
        predictor = ConditionalAncestralSamplingPredictor(*predictor_params)
    elif config.predictor_model == "CondReverseDiffusion":
        predictor = ConditionalReverseDiffusionSamplingPredictor(*predictor_params)
    elif config.predictor_model == "ProbODE":
        predictor = ConditionalProbODESamplingPredictor(*predictor_params)

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
    paths = []
    means = []
    vars = []
    prev_path = torch.zeros(size=(data_shape[0], 1))
    for t in range(config.ts_length):
        print("Sampling at real time {}\n".format(t + 1))
        if t == 0:
            features = torch.zeros(size=(data_shape[0], 1, config.mkv_blnk * config.ts_dims)).to(device)
        else:
            if "fBn" not in config.data_path:
                past = [torch.zeros_like(paths[0]) for _ in range(max(0, config.mkv_blnk - t))] + paths
                past = torch.stack(past, dim=1)
                past = past.cumsum(dim=1)
                assert (past.shape == (data_shape[0], max(1, t), config.ts_dims, 1))
                features = past[:, -config.mkv_blnk:, :].reshape(
                    (data_shape[0], 1, config.mkv_blnk * config.ts_dims, 1)).squeeze(-1)
                assert (features.shape == (data_shape[0], 1, config.mkv_blnk * config.ts_dims))
            else:
                past = [torch.zeros_like(paths[0]) for _ in range(max(0, config.mkv_blnk - t))] + paths[
                                                                                                  -config.mkv_blnk:]
                features = torch.stack(past, dim=1).reshape(
                    (data_shape[0], 1, config.mkv_blnk * config.ts_dims, 1)).squeeze(-1)
        samples, mean, var = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device,
                                            feature=features,
                                            early_stop_idx=config.early_stop_idx, ts_step=1. / config.ts_length,
                                            param_time=config.param_time, prev_path=prev_path)
        # Samples are size (BatchSize, 1, TimeSeriesDimension)
        if t != 0:
            est_mean = mean * config.ts_length
            assert (est_mean.shape == prev_path.shape)
            print("Estimated drift {}\n".format(est_mean))
            print("Expected drift {}\n".format(-config.mean_rev * prev_path))
            print(torch.mean(est_mean), torch.std(est_mean), torch.mean(prev_path), torch.std(prev_path),
                  torch.mean(-config.mean_rev * prev_path), torch.std(-config.mean_rev * prev_path))
        assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
        paths.append(samples.detach())
        means.append(mean.detach())
        vars.append(var.detach())
        prev_path = torch.concat(paths, dim=1).squeeze(dim=2).sum(axis=1).unsqueeze(-1)
    final_paths = np.atleast_2d(torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2).numpy())
    conditional_means = np.atleast_2d(torch.concat(means, dim=1).cpu().numpy())
    conditional_vars = np.atleast_2d(torch.concat(vars, dim=1).cpu().numpy())
    assert (final_paths.shape == conditional_means.shape == conditional_vars.shape)
    return final_paths, conditional_means, conditional_vars


def prepare_recursive_scoreModel_data(data: Union[np.ndarray, torch.Tensor], batch_size: int,
                                      config: ConfigDict) -> DataLoader:
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
                                             scoreModel: Union[ConditionalMarkovianTSPostMeanScoreMatching, ConditionalLSTMTSPostMeanScoreMatching],
                                             trainClass: Union[ConditionalStbleTgtMarkovianPostMeanDiffTrainer, ConditionalMarkovianPostMeanDiffTrainer, ConditionalStbleTgtMarkovianPostMeanDiffTrainer]) -> None:
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
    if isinstance(trainClass, type) and (issubclass(trainClass, ConditionalStbleTgtMarkovianPostMeanDiffTrainer)or issubclass(trainClass, ConditionalStbleTgtMarkovianScoreDiffTrainer)):
        trainLoader = prepare_recursive_scoreModel_data(data=data, batch_size=config.ref_batch_size, config=config)
    else:
        trainLoader = prepare_recursive_scoreModel_data(data=data, batch_size=config.batch_size, config=config)

    # Define optimiser
    print(f"Learning Rate\n: {config.lr}")
    optimiser = torch.optim.Adam((scoreModel.parameters()), lr=config.lr)
    # Define trainer
    train_eps, end_diff_time, max_diff_steps, checkpoint_freq = config.train_eps, config.end_diff_time, config.max_diff_steps, config.save_freq
    print(isinstance(config.initState, float))
    if isinstance(config.initState, float):
        init_state = torch.Tensor([config.initState])
    else:
        init_state = torch.Tensor(config.initState)

    if isinstance(trainClass, type) and issubclass(trainClass, ConditionalMarkovianPostMeanDiffTrainer):
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                             to_weight=config.weightings,
                             loss_factor=config.loss_factor,
                             hybrid_training=config.hybrid, init_state=init_state, deltaT=config.deltaT)
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
    elif isinstance(trainClass, type) and issubclass(trainClass, ConditionalStbleTgtMarkovianPostMeanDiffTrainer):
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                             to_weight=config.weightings,
                             hybrid_training=config.hybrid, loss_factor=config.loss_factor,
                             init_state=init_state, deltaT=config.deltaT)

        # Start training
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path, batch_size=config.batch_size, config=config)
    elif isinstance(trainClass, type) and issubclass(trainClass, ConditionalStbleTgtMarkovianScoreDiffTrainer):
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                             to_weight=config.weightings,
                             hybrid_training=config.hybrid, loss_factor=config.loss_factor,
                             init_state=init_state, deltaT=config.deltaT)

        # Start training
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path,
                      batch_size=config.batch_size, config=config)
    elif isinstance(trainClass, type) and issubclass(trainClass, ConditionalStbleTgtLSTMPostMeanDiffTrainer):
        # Post Mean LSTM
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                             to_weight=config.weightings,
                             hybrid_training=config.hybrid, loss_factor=config.loss_factor,
                             init_state=init_state, deltaT=config.deltaT)

        # Start training
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path,
                      batch_size=config.batch_size, config=config)
    else:
        raise RuntimeError("Invalid Diffusion Training Class\n")
