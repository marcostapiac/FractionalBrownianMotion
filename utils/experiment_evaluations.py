import pickle
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict
from tqdm import tqdm

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTM import DiscriminativeLSTM
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTM import PredictiveLSTM
from src.evaluation_pipeline.data_processing import test_predLSTM, test_discLSTM, train_and_save_predLSTM, \
    train_and_save_discLSTM
from src.generative_modelling.data_processing import reverse_sampling, train_and_save_diffusion_model
from src.generative_modelling.models.ClassOUSDEDiffusion import OUSDEDiffusion
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import generate_circles, generate_sine_dataset
from utils.math_functions import chiSquared_test, reduce_to_fBn, compute_fBm_cov, permutation_test, \
    energy_statistic, MMD_statistic, generate_fBm, compute_circle_proportions, generate_fBn, estimate_hurst, \
    compute_pvals, compute_fBn_cov


def prepare_sines_experiment(diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                             scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                             rng: np.random.Generator, config: ConfigDict) -> Union[NaiveMLP, TimeSeriesScoreMatching]:
    """
    Helper function to train and / or load necessary models for sines experiments
        :param diffusion: Diffusion model
        :param scoreModel: Score network
        :param rng: Default random number generator
        :param config: ML experiment configuration file
        :return: Trained score network
    """
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_sine_dataset(T=config.ts_length, S=training_size, rng=rng)
            np.save(config.data_path, data)
        data = data.cumsum(axis=1)[:training_size, :]
        train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))

    if config.test_pred_lstm:
        try:
            torch.load(config.pred_lstm_trained_path + "_NEp" + str(config.pred_lstm_max_epochs))
        except FileNotFoundError as e:
            print("Error {}; training predictive LSTM\n".format(e))
            pred = PredictiveLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in pred.parameters() if p.requires_grad), 2000000)
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.ts_length),
                                         config=config)
            train_and_save_predLSTM(data=synthetic.cpu().numpy(), config=config, model=pred)
    if config.test_disc_lstm:
        try:
            torch.load(config.disc_lstm_trained_path + "_NEp" + str(config.disc_lstm_max_epochs))
        except FileNotFoundError as e:
            print("Error {}; training discriminative LSTM\n".format(e))
            disc = DiscriminativeLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in disc.parameters() if p.requires_grad), 2000000)
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.ts_length),
                                         config=config)
            original = generate_sine_dataset(T=config.ts_length, S=config.dataSize, rng=rng)
            train_and_save_discLSTM(org_data=original, synth_data=synthetic.cpu().numpy(), config=config, model=disc)
    return scoreModel


def run_sines_experiment(dataSize: int, diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                         scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                         rng: np.random.Generator, config: ConfigDict) -> None:
    """
    Run sines experiment
        :param dataSize: Size of output data
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param rng: Default random number generator
        :param config: ML configuration file
        :return: None
    """
    agg_dict = {i + 1: None for i in range(config.num_runs)}
    for j in range(1, config.num_runs + 1):
        exp_dict = {key: None for key in config.exp_keys}
        try:
            assert (config.train_eps <= config.sample_eps)
            sines_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                             data_shape=(dataSize, config.ts_length),
                                             config=config)
        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")

        true_samples = generate_sine_dataset(T=config.ts_length, S=dataSize, rng=rng)
        exp_dict = evaluate_sines_performance(true_samples, sines_samples.cpu().numpy(), rng=rng, config=config,
                                              exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip", index=True)
    print(pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip",
                      index_col=[0]))


def evaluate_sines_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator,
                               config: ConfigDict, exp_dict: dict) -> dict:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples
        :param generated_samples: Final reverse-time diffusion samples
        :param rng: Default random number generator
        :param config: Configuration dictionary for experiment
        :param exp_dict: Dictionary storing current experiment results
        :return: None
    """
    true_mean = np.mean(true_samples, axis=0)
    print("True Data Sample Mean :: ", true_mean)
    gen_mean = np.mean(generated_samples, axis=0)
    print("Generated Data Sample Mean :: ", gen_mean)
    exp_dict[config.exp_keys[0]] = 100 * np.mean(np.abs((gen_mean - true_mean) / true_mean))

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)

    exp_dict[config.exp_keys[1]] = 100 * np.mean(np.abs((gen_cov - true_cov) / true_cov))

    # Evaluate marginal distributions
    ps = compute_pvals(true_samples, generated_samples)
    exp_dict[config.exp_keys[2]] = ps

    if config.test_pred_lstm:
        # Predictive LSTM test
        org, synth = test_predLSTM(original_data=true_samples, synthetic_data=generated_samples,
                                   model=PredictiveLSTM(ts_dim=1),
                                   config=config)
        exp_dict[config.exp_keys[3]] = org
        exp_dict[config.exp_keys[4]] = synth

    if config.test_disc_lstm:
        # Discriminative LSTM test
        org, synth = test_discLSTM(original_data=true_samples, synthetic_data=generated_samples,
                                   model=DiscriminativeLSTM(ts_dim=1),
                                   config=config)
        exp_dict[config.exp_keys[5]] = org
        exp_dict[config.exp_keys[6]] = synth

    if config.permute_test:
        # Permutation test for kernel statistic
        test_L = min(2000, true_samples.shape[0])
        print("MMD Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                             num_permutations=1000)))
        # Permutation test for energy statistic
        print("Energy Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                             num_permutations=1000)))
    return exp_dict


def prepare_fBm_experiment(diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                           scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                           rng: np.random.Generator, config: ConfigDict) -> Union[NaiveMLP, TimeSeriesScoreMatching]:
    """
    Helper function to train and / or load necessary models for fBm experiments
        :param diffusion: Diffusion model
        :param scoreModel: Score network
        :param rng: Default random number generator
        :param config: ML experiment configuration file
        :return: Trained score network
    """
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = int(min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000))
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fBn(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                                H=config.hurst)
            np.save(config.data_path, data)
        if config.isfBm:
            data = data.cumsum(axis=1)[:training_size, :]
        else:
            data = data[:training_size, :]
        train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))

    if config.test_pred_lstm:
        try:
            torch.load(config.pred_lstm_trained_path + "_NEp" + str(config.pred_lstm_max_epochs))
        except FileNotFoundError as e:
            print("Error {}; training predictive LSTM\n".format(e))
            pred = PredictiveLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in pred.parameters() if p.requires_grad), 2000000)
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.ts_length),
                                         config=config)
            train_and_save_predLSTM(data=synthetic.cpu().numpy(), config=config, model=pred)
    if config.test_disc_lstm:
        try:
            torch.load(config.disc_lstm_trained_path + "_NEp" + str(config.disc_lstm_max_epochs))
        except FileNotFoundError as e:
            print("Error {}; training discriminative LSTM\n".format(e))
            disc = DiscriminativeLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in disc.parameters() if p.requires_grad), 2000000)
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.ts_length),
                                         config=config)
            original = generate_fBm(H=config.hurst, T=config.ts_length, S=dataSize, rng=rng,
                                    isUnitInterval=config.isUnitInterval)
            train_and_save_discLSTM(org_data=original, synth_data=synthetic.cpu().numpy(), config=config, model=disc)
    return scoreModel


def run_fBm_experiment(dataSize: int, diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                       scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                       rng: np.random.Generator, config: ConfigDict) -> None:
    """
    Run fBm experiment
        :param dataSize: Size of output data
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param rng: Default random number generator
        :param config: ML configuration file
        :return: None
    """
    agg_dict = {i + 1: None for i in range(config.num_runs)}
    for j in range(1, config.num_runs + 1):
        exp_dict = {key: None for key in config.exp_keys}
        try:
            assert (config.train_eps <= config.sample_eps)
            synth_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                             data_shape=(dataSize, config.ts_length),
                                             config=config)
        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")
        if config.isfBm:
            true_samples = generate_fBm(H=config.hurst, T=config.ts_length, S=dataSize,
                                        isUnitInterval=config.isUnitInterval)
        else:
            true_samples = generate_fBn(H=config.hurst, T=config.ts_length, S=dataSize,
                                        isUnitInterval=config.isUnitInterval)
        exp_dict = evaluate_fBm_performance(true_samples, synth_samples.cpu().numpy(), rng=rng, config=config,
                                            exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip",
              index=True)
    print(pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip",
                      index_col=[0]))


def evaluate_fBm_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator,
                             config: ConfigDict, exp_dict: dict) -> dict:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples of fractional Brownian motion (or its increments)
        :param generated_samples: Final reverse-time diffusion samples
        :param rng: Default random number generator
        :param config: Configuration dictionary for experiment
        :param exp_dict: Dictionary storing current experiment results
        :return: None
    """
    true_mean = np.mean(true_samples, axis=0)
    print("True Data Sample Mean :: ", true_mean)
    gen_mean = np.mean(generated_samples, axis=0)
    print("Generated Data Sample Mean :: ", gen_mean)
    exp_dict.update({config.exp_keys[0]: 100 * np.mean(np.abs((gen_mean - true_mean) / true_mean))})

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                                isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)
    exp_dict[config.exp_keys[1]] = 100 * np.mean(np.abs((gen_cov - expec_cov) / expec_cov))

    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of exact fractional Brownian noise
    c2 = chiSquared_test(T=config.ts_length, H=config.hurst, samples=reduce_to_fBn(true_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    exp_dict[config.exp_keys[2]] = c2[0]
    exp_dict[config.exp_keys[3]] = c2[2]
    exp_dict[config.exp_keys[4]] = c2[1]
    print(
        "Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], np.sum(c2[1]),
                                                                                                   c2[2]))
    # Chi-2 test for joint distribution of synthetic fractional Brownian noise
    c2 = chiSquared_test(T=config.ts_length, H=config.hurst,
                         samples=reduce_to_fBn(generated_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0],
                                                                                                       np.sum(c2[1]),
                                                                                                       c2[2]))
    exp_dict[config.exp_keys[5]] = c2[1]

    # Evaluate marginal distributions
    ps = compute_pvals(true_samples, generated_samples)
    exp_dict[config.exp_keys[6]] = ps

    if config.test_pred_lstm:
        # Predictive LSTM test
        org, synth = test_predLSTM(original_data=true_samples, synthetic_data=generated_samples,
                                   model=PredictiveLSTM(ts_dim=1),
                                   config=config)
        exp_dict[config.exp_keys[7]] = org
        exp_dict[config.exp_keys[8]] = synth

    if config.test_disc_lstm:
        # Discriminative LSTM test
        org, synth = test_discLSTM(original_data=true_samples, synthetic_data=generated_samples,
                                   model=DiscriminativeLSTM(ts_dim=1),
                                   config=config)
        exp_dict[config.exp_keys[9]] = org
        exp_dict[config.exp_keys[10]] = synth

    if config.permute_test:
        # Permutation test for kernel statistic
        test_L = min(2000, true_samples.shape[0])
        print("MMD Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                             num_permutations=1000)))
        # Permutation test for energy statistic
        print("Energy Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                             num_permutations=1000)))

    exp_dict = estimate_hurst(true=true_samples, synthetic=generated_samples, exp_dict=exp_dict, S=S, config=config)

    return exp_dict


def prepare_circle_experiment(diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                              scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], config: ConfigDict) -> Union[
    NaiveMLP, TimeSeriesScoreMatching]:
    """
        Helper function to train and / or load necessary models for fBm experiments
            :param diffusion: Diffusion model
            :param scoreModel: Score network
            :param config: ML experiment configuration file
            :return: Trained score network
        """
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_circles(S=training_size, noise=config.cnoise)
            np.save(config.data_path, data)  # TODO is this the most efficient way
        train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(config.max_epochs)))
    return scoreModel


def run_circle_experiment(dataSize: int, diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                          scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                          config: ConfigDict) -> None:
    """
    Run circle experiment
        :param dataSize: Size of output data
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param config: ML configuration file
        :return: None
    """
    agg_dict = {i + 1: None for i in range(config.num_runs)}
    for j in range(1, config.num_runs + 1):
        exp_dict = {key: None for key in config.exp_keys}
        try:
            assert (config.train_eps <= config.sample_eps)
            circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                              data_shape=(dataSize, config.ts_length), config=config)

        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")

        true_samples = generate_circles(S=dataSize, noise=config.cnoise)
        exp_dict = evaluate_circle_performance(true_samples, circle_samples.cpu().numpy(), config=config,
                                               exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip", index=True)
    print(pd.read_csv(config.experiment_path + "_NEp{}.csv.gzip".format(config.max_epochs), compression="gzip",
                      index_col=[0]))


def evaluate_circle_performance(true_samples: np.ndarray, generated_samples: np.ndarray, config: ConfigDict,
                                exp_dict: dict) -> dict:
    """
    Compute various quantitative and qualitative metrics on final reverse-time diffusion samples for circle dataset
        :param true_samples: Exact samples from circle dataset
        :param generated_samples: Final reverse-time diffusion samples
        :param config: Configuration dictionary for experiment
        :param exp_dict: Dictionary storing current experiment results
        :return: Dictionary with experiment results
    """
    true_mean = np.mean(true_samples, axis=0)
    print("True Data Sample Mean :: ", true_mean)
    gen_mean = np.mean(generated_samples, axis=0)
    print("Generated Data Sample Mean :: ", gen_mean)
    exp_dict[config.exp_keys[0]] = 100 * np.mean(np.abs((gen_mean - true_mean) / true_mean))

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)
    exp_dict[config.exp_keys[1]] = 100 * np.mean(np.abs((gen_cov - true_cov) / true_cov))

    ps = compute_pvals(true_samples, generated_samples)
    exp_dict[config.exp_keys[2]] = ps

    true_prop, gen_prop = compute_circle_proportions(true_samples, generated_samples)
    exp_dict[config.exp.keys[3]] = true_prop
    exp_dict[config.exp.keys[4]] = gen_prop

    if config.permute_test:
        # Permutation test for kernel statistic
        test_L = min(2000, true_samples.shape[0])
        print("MMD Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                             num_permutations=1000)))
        # Permutation test for energy statistic
        print("Energy Permutation test: p-value {}".format(
            permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                             num_permutations=1000)))
    return exp_dict


def run_fBm_score_error_experiment(dataSize: int,
                                   diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                                   scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], rng: np.random.Generator,
                                   config: ConfigDict) -> torch.Tensor:
    """
        Visualise the error between score
            :param dataSize: Size of output data
            :param diffusion: Diffusion model
            :param scoreModel: Trained score network
            :param rng: Default random number generator
            :param config: ML configuration file
            :return: Tensor of errors over time and space
        """
    try:
        assert (config.train_eps <= config.sample_eps)
    except AssertionError as e:
        raise ValueError(
            "Error {};Final time during sampling should be at least as large as final time during training\n".format(e))
    try:
        assert (config.predictor_model == "ancestral")
    except AssertionError as e:
        print("Error {}; only ancestral sampling is supported currently\n".format(e))

    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    # Compute covariance function to compute exact score afterwards
    data_cov = torch.from_numpy(
        compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                        isUnitInterval=config.isUnitInterval)).to(
        torch.float32).to(device) if config.isfBm else torch.from_numpy(
        compute_fBn_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                        isUnitInterval=config.isUnitInterval)).to(
        torch.float32).to(device)

    # Placeholder
    errors = torch.zeros(size=(config.max_diff_steps, config.ts_length))

    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps).to(
        device)
    x = diffusion.prior_sampling(shape=(dataSize, config.ts_length)).to(device)  # Move to correct device
    scoreModel.eval()
    scoreModel.to(device)
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Score Error Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)

        # Obtain required diffusion parameters
        pred_score, drift, diffusion_param = diffusion.get_ancestral_sampling(x, t=timesteps[
                                                                                       diff_index.long()] * torch.ones(
            (x.shape[0], 1)).to(device), score_network=scoreModel, diff_index=diff_index,
                                                                              max_diff_steps=config.max_diff_steps)
        eff_time = diffusion.get_eff_times(diff_times=timesteps[diff_index.long()])
        if isinstance(diffusion, VESDEDiffusion):
            inv_cov = -torch.linalg.inv(eff_time * torch.eye(config.ts_length).to(device) + data_cov)
        else:
            inv_cov = -torch.linalg.inv(
                (1. - torch.exp(-eff_time)) * torch.eye(config.ts_length).to(device) + torch.exp(-eff_time) * data_cov)

        exp_score = (inv_cov @ x.T).T

        errors[config.max_diff_steps - 1 - i, :] = torch.pow(torch.linalg.norm(pred_score - exp_score, ord=2, axis=0),
                                                             2).detach().cpu()

        # One-step reverse-time SDE
        x = drift + diffusion_param * torch.randn_like(x)

    return errors


def run_fBm_backward_drift_experiment(dataSize: int,
                                      diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                                      scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], rng: np.random.Generator,
                                      config: ConfigDict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Visualise the error between drifts
        :param dataSize: Size of output data
        :param diffusion: Diffusion model
        :param scoreModel: Trained score network
        :param rng: Default random number generator
        :param config: ML configuration file
        :return: Tensor of errors over time and space
    """
    try:
        assert (config.train_eps <= config.sample_eps)
    except AssertionError as e:
        raise ValueError(
            "Error {}; Final time during sampling should be at least as large as final time during training\n".format(
                e))
    try:
        assert (config.predictor_model == "ancestral")
    except AssertionError as e:
        print("Error {}; only ancestral sampling is supported currently\n".format(e))

    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    if config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)

    # Placeholder
    drift_errors = torch.zeros(size=(config.max_diff_steps, config.ts_length))
    score_only_errors = torch.zeros(size=(config.max_diff_steps, config.ts_length))

    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps).to(
        device)
    x = diffusion.prior_sampling(shape=(dataSize, config.ts_length)).to(device)  # Move to correct device

    scoreModel.eval()
    scoreModel.to(device)
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Drift Error Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        max_diff_steps = torch.Tensor([config.max_diff_steps]).to(device)

        # Obtain required diffusion parameters
        _, pred_drift, diffusion_param = diffusion.get_ancestral_sampling(x,
                                                                          t=timesteps[diff_index.long()] * torch.ones(
                                                                              (x.shape[0], 1)).to(device),
                                                                          score_network=scoreModel,
                                                                          diff_index=diff_index,
                                                                          max_diff_steps=config.max_diff_steps)
        eff_time = diffusion.get_eff_times(diff_times=timesteps[diff_index.long()])
        if isinstance(diffusion, VESDEDiffusion):
            inv_cov = -torch.linalg.inv(eff_time * torch.eye(config.ts_length).to(device) + data_cov)
            score_only_pred_drift = pred_drift - x
            exp_score_only_drift = (diffusion.get_ancestral_drift_coeff(max_diff_steps=max_diff_steps,
                                                                        diff_index=diff_index)) * (inv_cov @ x.T).T

            exp_drift = x + exp_score_only_drift
        else:
            inv_cov = -torch.linalg.inv(
                (1. - torch.exp(-eff_time)) * torch.eye(config.ts_length).to(device) + torch.exp(-eff_time) * data_cov)
            beta_t = diffusion.get_discretised_beta(max_diff_steps - 1 - diff_index, max_diff_steps)
            score_only_pred_drift = pred_drift - x * (2. - torch.sqrt(1. - beta_t))
            exp_score_only_drift = (beta_t * (inv_cov @ x.T).T)
            exp_drift = x * (2. - torch.sqrt(1. - beta_t)) + exp_score_only_drift

        drift_errors[config.max_diff_steps - 1 - i, :] = torch.pow(
            torch.linalg.norm(pred_drift - exp_drift, ord=2, axis=0),
            2).detach().cpu()
        score_only_errors[config.max_diff_steps - 1 - i, :] = torch.pow(
            torch.linalg.norm(score_only_pred_drift - exp_score_only_drift, ord=2, axis=0),
            2).detach().cpu()

        # One-step reverse-time SDE
        x = pred_drift + diffusion_param * torch.randn_like(x)

    return drift_errors, score_only_errors


def run_fBm_score(dataSize: int, dim_pair: torch.Tensor, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                  diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion], folderPath: str, gifPath: str,
                  rng: np.random.Generator, config: ConfigDict) -> torch.Tensor:
    """
    Run reverse-time diffusion and plot scatter plot for neighbouring dimensions and compare with theoretical contours
        :param dataSize: Size of output data
        :param dim_pair: Vector of dimensions of interest
        :param scoreModel: Trained score network
        :param diffusion: Diffusion model
        :param folderPath: Path to folder to save images
        :param gifPath: Path to save GIF
        :param rng: Default random number generator
        :param config: ML configuration file
        :return: None
    """
    from utils.plotting_functions import plot_and_save_diffused_fBm_snapshot
    try:
        assert (dim_pair.shape[0] == 2)
    except AssertionError:
        raise ValueError("You can only choose a pair of dimensions to plot\n")
    try:
        assert (config.train_eps <= config.sample_eps)
    except AssertionError as e:
        raise ValueError(
            "Error {}; Final time during sampling should be at least as large as final time during training\n".format(
                e))
    try:
        assert (config.predictor_model == "ancestral")
    except AssertionError as e:
        print("Error {}; only ancestral sampling is supported currently\n".format(e))
    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    dim_pair = dim_pair.to(device)
    x = diffusion.prior_sampling(shape=(dataSize, config.ts_length)).to(device)  # Move to correct device

    if config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)

    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps).to(
        device)
    scoreModel.to(device)
    scoreModel.eval()
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Backward Diffusion Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        eff_time = diffusion.get_eff_times(diff_times=timesteps[diff_index.long()])

        if isinstance(diffusion, VESDEDiffusion):
            diffType = "VESDE"
            cov = eff_time * torch.eye(config.ts_length).to(device) + data_cov
        else:
            diffType = "VPSDE"
            cov = (1. - torch.exp(-eff_time)) * torch.eye(config.ts_length).to(device) + torch.exp(-eff_time) * data_cov

        pred_score, drift, diffusion_param = diffusion.get_ancestral_sampling(x, t=timesteps[
                                                                                       diff_index.long()] * torch.ones(
            (x.shape[0], 1)).to(device), score_network=scoreModel, diff_index=diff_index,
                                                                              max_diff_steps=config.max_diff_steps)
        if i % config.gif_save_freq == 0 or i == (config.max_diff_steps - 1):
            save_path = folderPath + gifPath + "_diffIndex_{}.png".format(i + 1)
            xlabel = "fBm Dimension {}".format(dim_pair[0] + 1)
            ylabel = "fBm Dimension {}".format(dim_pair[1] + 1)
            plot_title = "Rev-Time {} samples $T={}$ at time {}".format(diffType, config.ts_length, round((
                                                                                                                  i + 1) / config.max_diff_steps,
                                                                                                          5))
            paired_cov = torch.index_select(torch.index_select(cov, dim=0, index=dim_pair), dim=1, index=dim_pair).to(
                device)
            plot_and_save_diffused_fBm_snapshot(samples=x[:, dim_pair].cpu(), cov=paired_cov.cpu(), save_path=save_path,
                                                x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

        x = drift + diffusion_param * torch.randn_like(x)
    return x


def run_fBm_scatter_matrix(dataSize: int, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                           diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion], folderPath: str,
                           gifPath: str,
                           rng: np.random.Generator, config: ConfigDict) -> None:
    from utils.plotting_functions import my_pairplot
    """
        Run reverse-time diffusion and plot scatter plot for neighbouring dimensions and compare with theoretical contours
            :param dataSize: Size of output data
            :param scoreModel: Trained score network
            :param diffusion: Diffusion model
            :param folderPath: Path to folder to save images
            :param gifPath: Path to save GIF
            :param rng: Default random number generator
            :param config: ML configuration file
            :return: None
        """
    try:
        assert (config.train_eps <= config.sample_eps)
    except AssertionError as e:
        raise ValueError(
            "Error {}; Final time during sampling should be at least as large as final time during training\n".format(
                e))
    try:
        assert (config.predictor_model == "ancestral")
    except AssertionError as e:
        print("Error {}; only ancestral sampling is supported currently\n".format(e))
    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    x = diffusion.prior_sampling(shape=(dataSize, config.ts_length)).to(device)  # Move to correct device

    if config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.ts_length,
                            isUnitInterval=config.isUnitInterval)).to(
            torch.float32).to(device)

    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps).to(
        device)
    scoreModel.to(device)
    scoreModel.eval()
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Backward Diffusion Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        eff_time = diffusion.get_eff_times(diff_times=timesteps[diff_index.long()])

        if isinstance(diffusion, VESDEDiffusion):
            diffType = "VESDE"
            cov = eff_time * torch.eye(config.ts_length).to(device) + data_cov
        else:
            diffType = "VPSDE"
            cov = (1. - torch.exp(-eff_time)) * torch.eye(config.ts_length).to(device) + torch.exp(-eff_time) * data_cov

        pred_score, drift, diffusion_param = diffusion.get_ancestral_sampling(x, t=timesteps[
                                                                                       diff_index.long()] * torch.ones(
            (x.shape[0], 1)).to(device), score_network=scoreModel, diff_index=diff_index,
                                                                              max_diff_steps=config.max_diff_steps)
        if (i >= config.idx_start_save) and (i % config.gif_save_freq == 0 or i == (config.max_diff_steps - 1)):
            row_vars = config.row_idxs
            col_vars = config.col_idxs
            save_path = folderPath + gifPath + "_dI_{}.png".format(i + 1)
            title = "Rev-Time {} samples $T={}$ at time {}".format(diffType, config.ts_length,
                                                                   round((i + 1) / config.max_diff_steps, 5))
            my_pairplot(x.cpu(), row_idxs=row_vars, col_idxs=col_vars, cov=cov.cpu(), image_path=save_path,
                        suptitle=title)

        x = drift + diffusion_param * torch.randn_like(x)
    return x


def run_fBm_perfect_score(dataSize: int, dim_pair: torch.Tensor,
                          diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion], folderPath: str,
                          gifPath: str,
                          rng: np.random.Generator, perfect_config: ConfigDict) -> None:
    from utils.plotting_functions import plot_and_save_diffused_fBm_snapshot
    """
    Run reverse-time diffusion under perfect VESDE score knowledge and plot scatter plot for neighbouring dimensions
        :param dataSize: Size of output data
        :param dim_pair: Vector of dimensions of interest
        :param diffusion: Diffusion model
        :param folderPath: Path to folder to save images
        :param gifPath: Path to save GIF
        :param rng: Default random number generator
        :param perfect_config: ML configuration file
        :return: None
    """
    try:
        assert (dim_pair.shape[0] == 2)
    except AssertionError:
        raise ValueError("You can only choose a pair of dimensions to plot\n")
    if perfect_config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")
    # NOTE: No need to diffuse whole sample since the score is set correctly manually
    x = diffusion.prior_sampling(shape=(dataSize, dim_pair.shape[0])).to(device)  # Move to correct device

    if perfect_config.isfBm:
        data_cov = torch.from_numpy(
            compute_fBm_cov(FractionalBrownianNoise(H=perfect_config.hurst, rng=rng), T=perfect_config.ts_length,
                            isUnitInterval=perfect_config.isUnitInterval)).to(
            torch.float32).to(device)
    else:
        data_cov = torch.from_numpy(
            compute_fBn_cov(FractionalBrownianNoise(H=perfect_config.hurst, rng=rng), T=perfect_config.ts_length,
                            isUnitInterval=perfect_config.isUnitInterval)).to(
            torch.float32).to(device)

    data_cov = torch.index_select(torch.index_select(data_cov, dim=0, index=dim_pair), dim=1, index=dim_pair).to(device)

    for i in tqdm(iterable=(range(0, perfect_config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Backward Diffusion Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        max_diff_steps = torch.Tensor([perfect_config.max_diff_steps]).to(device)
        eff_time = diffusion.get_eff_times(diff_times=(max_diff_steps - 1 - diff_index) / (max_diff_steps - 1))

        if isinstance(diffusion, VESDEDiffusion):
            diffType = "VESDE"
            cov = eff_time * torch.eye(dim_pair.shape[0]).to(device) + data_cov
        else:
            diffType = "VPSDE"
            cov = (1. - torch.exp(-eff_time)) * torch.eye(dim_pair.shape[0]).to(device) + torch.exp(
                -eff_time) * data_cov

        # Compute exact score
        exp_score = (-torch.linalg.inv(cov) @ x.T).T
        if perfect_config.predictor_model == "ancestral":
            drift = diffusion.get_ancestral_drift(x=x, pred_score=exp_score, diff_index=diff_index,
                                                  max_diff_steps=max_diff_steps)
            diffusion_param = diffusion.get_ancestral_diff(diff_index=diff_index, max_diff_steps=max_diff_steps)
        else:
            raise ValueError("Alternative to ancestral sampling has not been implemented\n")
        if i % perfect_config.gif_save_freq == 0 or i == (perfect_config.max_diff_steps - 1):
            save_path = folderPath + gifPath + "_diffIndex_{}.png".format(i + 1)
            xlabel = "fBm Dimension {}".format(dim_pair[0] + 1)
            ylabel = "fBm Dimension {}".format(dim_pair[1] + 1)
            plot_title = "Rev-Time {} samples $T={}$ at time {}".format(diffType, perfect_config.ts_length, round((
                                                                                                                          i + 1) / perfect_config.max_diff_steps,
                                                                                                                  5))
            plot_and_save_diffused_fBm_snapshot(samples=x.cpu(), cov=cov.cpu(), save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

        x = drift + diffusion_param * torch.randn_like(x)
    return x
