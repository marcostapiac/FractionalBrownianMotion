from typing import Union

import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict

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
    energy_statistic, MMD_statistic, generate_fBm, compute_circle_proportions, generate_fBn
from utils.plotting_functions import plot_final_diff_marginals, plot_dataset, \
    plot_diffCov_heatmap, plot_tSNE


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
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except FileNotFoundError as e:
        print("No valid trained model found; proceeding to training\n")
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except FileNotFoundError as e:
            print("Generating synthetic data\n")
            training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)
            data = generate_sine_dataset(T=config.timeDim, S=training_size, rng=rng)
            np.save(config.data_path, data)
        finally:
            data = data.cumsum(axis=1)
            train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))

    if config.test_pred_lstm:
        try:
            torch.load(config.pred_lstm_trained_path)
        except FileNotFoundError as e:
            pred = PredictiveLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in pred.parameters() if p.requires_grad), 2000000) // 10
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.timeDim),
                                         config=config)
            train_and_save_predLSTM(data=synthetic.cpu().numpy(), config=config, model=pred)
    if config.test_disc_lstm:
        try:
            torch.load(config.disc_lstm_trained_path)
        except FileNotFoundError as e:
            disc = DiscriminativeLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in disc.parameters() if p.requires_grad), 2000000) // 10
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.timeDim),
                                         config=config)
            original = generate_sine_dataset(T=config.timeDim, S=config.dataSize, rng=rng)
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
                                             data_shape=(dataSize, config.timeDim),
                                             config=config)
        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")

        true_samples = generate_sine_dataset(T=config.timeDim, S=dataSize, rng=rng)
        exp_dict = evaluate_sines_performance(true_samples, sines_samples.cpu().numpy(), rng=rng, config=config,
                                              exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path, index=True)  # For reading, pd.read_csv(config.experiment_path, index_col=[0])
    print(pd.read_csv(config.experiment_path, index_col=[0]))


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
    exp_dict[config.exp_keys[0]] = np.mean(np.abs(gen_mean - true_mean) / true_mean)

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)

    exp_dict[config.exp_keys[1]] = np.mean(np.abs(gen_cov - true_cov) / true_cov)

    plot_diffCov_heatmap(true_cov, gen_cov, annot=config.annot, image_path=config.image_path + "_diffCov.png")
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    plot_tSNE(x=true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"],
              image_path=config.image_path + "_tSNE") \
        if config.timeDim > 2 else plot_dataset(true_samples, generated_samples,
                                                image_path=config.image_path + "_scatter.png")
    # Evaluate marginal distributions
    ps = plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.print_marginals,
                                   timeDim=config.timeDim, image_path=config.image_path)
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
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except FileNotFoundError as e:
        print("No valid trained model found; proceeding to training\n")
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except FileNotFoundError as e:
            print("Generating synthetic data\n")
            training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)
            data = generate_fBn(T=config.timeDim, S=training_size, H=config.hurst, rng=rng)
            np.save(config.data_path, data)
        finally:
            data = data.cumsum(axis=1)
            train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))

    if config.test_pred_lstm:
        try:
            torch.load(config.pred_lstm_trained_path)
        except FileNotFoundError as e:
            pred = PredictiveLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in pred.parameters() if p.requires_grad), 2000000) // 10
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.timeDim),
                                         config=config)
            train_and_save_predLSTM(data=synthetic.cpu().numpy(), config=config, model=pred)
    if config.test_disc_lstm:
        try:
            torch.load(config.disc_lstm_trained_path)
        except FileNotFoundError as e:
            disc = DiscriminativeLSTM(ts_dim=1)
            dataSize = min(10 * sum(p.numel() for p in disc.parameters() if p.requires_grad), 2000000) // 10
            synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                         data_shape=(dataSize, config.timeDim),
                                         config=config)
            original = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, rng=rng)
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
            fBm_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                           data_shape=(dataSize, config.timeDim),
                                           config=config)
        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")
        true_samples = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, rng=rng)
        exp_dict = evaluate_fBm_performance(true_samples, fBm_samples.cpu().numpy(), rng=rng, config=config, exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path, index=True)  # For reading, pd.read_csv(config.experiment_path, index_col=[0])
    print(pd.read_csv(config.experiment_path, index_col=[0]))


def evaluate_fBm_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator,config: ConfigDict, exp_dict: dict) -> dict:
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
    exp_dict.update({config.exp_keys[0] : np.mean(np.abs(gen_mean - true_mean) / true_mean)})
    
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                                isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)
    exp_dict[config.exp_keys[1]] = np.mean(np.abs(gen_cov - expec_cov) / expec_cov)
    #plot_diffCov_heatmap(expec_cov, gen_cov, annot=config.annot, image_path=config.image_path + "_diffCov.png")
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst, samples=reduce_to_fBn(true_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    exp_dict[config.exp_keys[2]] = c2[0]
    exp_dict[config.exp_keys[3]] = c2[2]
    exp_dict[config.exp_keys[4]] = c2[1]
    print("Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                     c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst,
                         samples=reduce_to_fBn(generated_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))
    exp_dict[config.exp_keys[5]] = c2[1]

    """plot_tSNE(x=true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"],
              image_path=config.image_path + "_tSNE") \
        if config.timeDim > 2 else plot_dataset(true_samples, generated_samples,
                                                image_path=config.image_path + "_scatter.png")"""

    # Evaluate marginal distributions
    ps = plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.print_marginals,
                                   timeDim=config.timeDim, image_path=config.image_path)
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
    return exp_dict


def prepare_circle_experiment(diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                              scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching], config: ConfigDict) -> Union[
    NaiveMLP, TimeSeriesScoreMatching]:
    """
        Helper function to train and / or load necessary models for fBm experiments
            :param diffusion: Diffusion model
            :param scoreModel: Score network
            :param rng: Default random number generator
            :param config: ML experiment configuration file
            :return: Trained score network
        """
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except FileNotFoundError as e:
        print("No valid trained model found; proceeding to training\n")
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except FileNotFoundError as e:
            print("Generating synthetic data\n")
            training_size = min(10 * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 2000000)
            data = generate_circles(S=training_size, noise=config.cnoise)
            np.save(config.data_path, data)  # TODO is this the most efficient way
        finally:
            train_and_save_diffusion_model(data=data, config=config, diffusion=diffusion, scoreModel=scoreModel)
            scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    return scoreModel


def run_circle_experiment(dataSize: int, diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
                          scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                          config: ConfigDict) -> dict:
    """
    Run circle experiment
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
            circle_samples = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                              data_shape=(dataSize, config.timeDim), config=config)

        except AssertionError:
            raise ValueError("Final time during sampling should be at least as large as final time during training")

        true_samples = generate_circles(S=dataSize, noise=config.cnoise)
        exp_dict = evaluate_circle_performance(true_samples, circle_samples.cpu().numpy(), config=config,
                                               exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path, index=True)
    print(pd.read_csv(config.experiment_path, index_col=[0]))


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
    exp_dict[config.exp_keys[0]] = np.mean(np.abs(gen_mean - true_mean) / true_mean)

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)
    exp_dict[config.exp_keys[1]] = np.mean(np.abs(gen_cov - true_cov) / true_cov)

    plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter.png")
    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=gen_cov, image_path=config.image_path + "_scatter.png")
    ps = plot_final_diff_marginals(true_samples, generated_samples, timeDim=2, print_marginals=config.print_marginals,
                                   image_path=config.image_path)
    exp_dict[config.exp_keys[2]] = ps

    true_prop, gen_prop = compute_circle_proportions(true_samples, generated_samples)
    exp_dict[config.exp.keys[3]] = true_prop
    exp_dict[config.exp.keys[4]] = gen_prop

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
