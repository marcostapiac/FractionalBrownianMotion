import os
import pickle
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from torchmetrics.aggregation import MeanMetric
from ml_collections import ConfigDict
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, DistributedSampler

from src.classes.ClassConditionalDiffTrainer import ConditionalDiffusionModelTrainer
from src.classes.ClassConditionalSDESampler import ConditionalSDESampler
from src.classes.ClassCorrector import VESDECorrector, VPSDECorrector
from src.classes.ClassPredictor import AncestralSamplingPredictor, EulerMaruyamaPredictor, \
    ConditionalAncestralSamplingPredictor
from src.classes.ClassSDESampler import SDESampler
from src.generative_modelling.data_processing import prepare_scoreModel_data
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import init_experiment, cleanup_experiment
from utils.experiment_evaluations import run_fBm_experiment
from utils.math_functions import generate_fBn, generate_fBm


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
                                    loss_aggregator=MeanMetric,
                                    snapshot_path=config.scoreNet_snapshot_path, device=device,
                                    train_eps=train_eps,
                                    end_diff_time=end_diff_time, max_diff_steps=max_diff_steps, to_weight=config.weightings, hybrid_training=config.hybrid)

    # Start training
    trainer.train(max_epochs=config.max_epochs, model_filename=config.scoreNet_trained_path)


def run_recursive_fBm_experiment(diffusion, scoreModel, rng, config):
    dataSize = config.dataSize
    synth_samples = recursive_reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                     data_shape=(dataSize, config.timeDim, 1),
                                     config=config)
    if config.isfBm:
        true_samples = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, isUnitInterval=config.isUnitInterval)
    else:
        true_samples = generate_fBn(H=config.hurst, T=config.timeDim, S=dataSize, isUnitInterval=config.isUnitInterval)

@record
def recursive_reverse_sampling(diffusion: VPSDEDiffusion,
                     scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], data_shape: Tuple[int, int, int],
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
    assert(config.predictor_model == "ancestral")
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
    sampler = ConditionalSDESampler(diffusion=diffusion, sample_eps=config.sample_eps, predictor=predictor, corrector=corrector)

    scoreModel.eval()
    h = torch.randn((1 * 2, data_shape[0], 40)).to(device)
    c = torch.randn((1 * 2, data_shape[0], 40)).to(device)
    samples = torch.randn(size=(data_shape[0], 1, data_shape[-1])).to(device)
    paths = []
    for t in range(config.timeDim):
        output, (h, c) = (scoreModel.rnn(samples, (h, c)))
        samples = sampler.sample(shape=(data_shape[0], data_shape[-1]), torch_device=device, feature=output, early_stop_idx=config.early_stop_idx)
        assert(samples.shape == (data_shape[0], 1, data_shape[-1]))
        paths.append(samples)
        if t == 2:
            print(torch.concat(paths, dim=1), torch.concat(paths, dim=1).shape)
    final_paths = torch.concat(paths)
    assert(final_paths.shape == data_shape)
    early_stop_idx = 0
    df = pd.DataFrame(final_paths)
    df.index = pd.MultiIndex.from_product(
        [["Early Stop {}".format(early_stop_idx), "Final Time Samples"], [i for i in range(config.dataSize)]])
    df.to_csv(config.experiment_path.replace("/results/",
                                             "/results/early_stopping/") + "_EStop{}_Nepochs{}.csv.gzip".format(
        early_stop_idx, config.max_epochs), compression="gzip")
    return final_paths



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

    #cleanup_experiment()

    run_recursive_fBm_experiment(diffusion=diffusion, scoreModel=scoreModel, rng=rng, config=config)
