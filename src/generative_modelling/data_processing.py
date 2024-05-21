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

from src.classes.ClassConditionalLSTMDiffTrainer import ConditionalLSTMDiffusionModelTrainer
from src.classes.ClassConditionalMarkovianDiffTrainer import ConditionalMarkovianDiffusionModelTrainer
from src.classes.ClassConditionalPostMeanLSTMDiffTrainer import ConditionalLSTMPostMeanDiffusionModelTrainer
from src.classes.ClassConditionalPostMeanMarkovianDiffTrainer import ConditionalPostMeanMarkovianDiffTrainer
from src.classes.ClassConditionalSDESampler import ConditionalSDESampler
from src.classes.ClassConditionalSignatureDiffTrainer import ConditionalSignatureDiffusionModelTrainer
from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassDiffTrainer import DiffusionModelTrainer
from src.classes.ClassPredictor import AncestralSamplingPredictor, \
    ConditionalAncestralSamplingPredictor, ConditionalReverseDiffusionSamplingPredictor, \
    ConditionalProbODESamplingPredictor
from src.classes.ClassSDESampler import SDESampler
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSScoreMatching import \
    ConditionalMarkovianTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalSignatureTSScoreMatching import \
    ConditionalSignatureTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTSScoreMatching import \
    ConditionalTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTSScoreMatching import \
    TSScoreMatching
from utils.math_functions import compute_sig_size


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
                                   scoreModel: Union[NaiveMLP, TSScoreMatching]) -> None:
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
                     scoreModel: Union[NaiveMLP, TSScoreMatching], data_shape: Tuple[int, int],
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


def compute_current_sig_feature(ts_time: int, device: Union[int, str], past_feat: torch.Tensor, basepoint: torch.Tensor,
                                latest_path: torch.Tensor, config: ConfigDict,
                                score_network: ConditionalSignatureTSScoreMatching,
                                full_path: torch.Tensor) -> torch.Tensor:
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
    """assert(len(basepoint.shape)==len(latest_path.shape)==3)
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
        curr_feat = curr_feat.unsqueeze(dim=1)
    else:
        curr_feat = increment_sig
    assert (curr_feat.shape == past_feat.shape)"""
    if ts_time == 0: full_path = torch.zeros_like(latest_path)
    expectsig = score_network.signet.forward(full_path, time_ax=torch.atleast_2d(
        torch.arange(1 * min(1, ts_time), ts_time + 1) / T).T, basepoint=True)[:, [-1], :]
    # assert((torch.abs(expectsig-curr_feat).squeeze(1).sum(dim=1).sum(dim=0)) <= 1e-15)
    return expectsig


@record
def recursive_signature_reverse_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalSignatureTSScoreMatching,
                                         data_shape: Tuple[int, int, int],
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
    with torch.no_grad():
        paths = [torch.zeros((data_shape[0], 1, data_shape[-1])).to(
            device)]  # Initial starting point (can be set to anything)
        output = torch.zeros((data_shape[0], 1, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1)).to(
            device)
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            output = compute_current_sig_feature(ts_time=t, device=device, past_feat=output,
                                                 basepoint=paths[max(t - 1, 0)], latest_path=paths[max(0, t)],
                                                 config=config, score_network=scoreModel,
                                                 full_path=torch.concat(paths[1 * min(1, t):], dim=1))
            samples = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=output,
                                     early_stop_idx=config.early_stop_idx)
            assert (samples.shape == (data_shape[0], 1, data_shape[-1]))
            paths.append(samples)
    final_paths = torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2)[:, 1:]
    assert (final_paths.shape == (data_shape[0], data_shape[1]))
    return np.atleast_2d(final_paths.numpy())


@record
def recursive_LSTM_reverse_sampling(diffusion: VPSDEDiffusion,
                                    scoreModel: ConditionalTSScoreMatching, data_shape: Tuple[int, int, int],
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
    prev_path = torch.zeros(size=(data_shape[0], 1, 1)).to(device)
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
            assert (est_mean.shape == prev_path.shape)
            print("Estimated drift {}\n".format(est_mean))
            if "fSin" in config.data_path:
                print("fSin\n")
                print("Expected drift {}\n".format(config.mean_rev * torch.sin(prev_path)))
                print(torch.mean(est_mean), torch.std(est_mean), torch.mean(prev_path), torch.std(prev_path),
                      torch.mean(config.mean_rev * torch.sin(prev_path)), torch.std(config.mean_rev * torch.sin(prev_path)))
            elif "fOU" in config.data_path:
                print("fOU\n")
                print("Expected drift {}\n".format(-config.mean_rev * (prev_path)))
                print(torch.mean(est_mean), torch.std(est_mean), torch.mean(prev_path), torch.std(prev_path),
                      torch.mean(-config.mean_rev * (prev_path)),
                      torch.std(-config.mean_rev * (prev_path)))
        paths.append(samples.detach())
        means.append(mean.detach())
        vars.append(var.detach())
        prev_path = torch.concat(paths, dim=1).sum(dim=1)
        print(prev_path.shape)
    final_paths = np.atleast_2d(torch.squeeze(torch.concat(paths, dim=1).cpu(), dim=2).numpy())
    conditional_means = np.atleast_2d(torch.concat(means, dim=1).cpu().numpy())
    conditional_vars = np.atleast_2d(torch.concat(vars, dim=1).cpu().numpy())
    assert (final_paths.shape == conditional_means.shape == conditional_vars.shape)
    return final_paths, conditional_means, conditional_vars


@record
def recursive_markovian_reverse_sampling(diffusion: VPSDEDiffusion,
                                         scoreModel: ConditionalTSScoreMatching,
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
                                             scoreModel: Union[
                                                 NaiveMLP, ConditionalTSScoreMatching, ConditionalTSScoreMatching, ConditionalMarkovianTSPostMeanScoreMatching, ConditionalMarkovianTSScoreMatching],
                                             trainClass: Union[ConditionalLSTMPostMeanDiffusionModelTrainer,
                                                               ConditionalLSTMDiffusionModelTrainer, ConditionalMarkovianDiffusionModelTrainer, ConditionalPostMeanMarkovianDiffTrainer, ConditionalSignatureDiffusionModelTrainer, DiffusionModelTrainer]) -> None:
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
        ts_type = "fOU" if "fOU" in config.data_path else "fBm"
        trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                             checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                             loss_aggregator=MeanMetric,
                             snapshot_path=config.scoreNet_snapshot_path, device=device,
                             train_eps=train_eps,
                             end_diff_time=end_diff_time, max_diff_steps=max_diff_steps, to_weight=config.weightings,
                             mkv_blnk=config.mkv_blnk, ts_data=ts_type,
                             hybrid_training=config.hybrid)
        # Start training
        trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
    except (AttributeError, KeyError, TypeError) as e:
        try:
            # Post Mean Markovian
            ts_type = "fOU" if "fOU" in config.data_path else "fBm"
            trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                 checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                 loss_aggregator=MeanMetric,
                                 snapshot_path=config.scoreNet_snapshot_path, device=device,
                                 train_eps=train_eps,
                                 end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                 to_weight=config.weightings,
                                 mkv_blnk=config.mkv_blnk, ts_data=ts_type, loss_factor=config.loss_factor,
                                 ts_time_diff=1 / config.ts_length,
                                 hybrid_training=config.hybrid)
            trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
        except (AttributeError, KeyError, TypeError) as e:
            try:
                # Signature
                assert (config.sig_trunc)
                trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                     checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                     loss_aggregator=MeanMetric,
                                     snapshot_path=config.scoreNet_snapshot_path, device=device,
                                     train_eps=train_eps,
                                     end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                     to_weight=config.weightings,
                                     hybrid_training=config.hybrid)
                # Start training
                trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path,
                              ts_dims=config.ts_dims)
            except (AttributeError, KeyError, TypeError) as e:
                # LSTM
                try:
                    trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                         checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                         loss_aggregator=MeanMetric,
                                         snapshot_path=config.scoreNet_snapshot_path, device=device,
                                         train_eps=train_eps,
                                         end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                         to_weight=config.weightings, ts_time_diff=1 / config.ts_length,
                                         loss_factor=config.loss_factor,
                                         hybrid_training=config.hybrid)

                    # Start training
                    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
                except (AttributeError, KeyError, TypeError) as e:
                    # Post Mean LSTM
                    trainer = trainClass(diffusion=diffusion, score_network=scoreModel, train_data_loader=trainLoader,
                                         checkpoint_freq=checkpoint_freq, optimiser=optimiser, loss_fn=torch.nn.MSELoss,
                                         loss_aggregator=MeanMetric,
                                         snapshot_path=config.scoreNet_snapshot_path, device=device,
                                         train_eps=train_eps,
                                         end_diff_time=end_diff_time, max_diff_steps=max_diff_steps,
                                         to_weight=config.weightings,
                                         hybrid_training=config.hybrid, loss_factor=config.loss_factor,
                                         ts_time_diff=1 / config.ts_length)

                    # Start training
                    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)
