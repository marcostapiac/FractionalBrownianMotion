import ast
import pickle
from typing import Union

import numpy as np
import pandas as pd
import scipy
import torch
from matplotlib import pyplot as plt
from ml_collections import ConfigDict
from scipy import stats
from scipy.stats import kstest, ks_2samp
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
    compute_pvals
from utils.plotting_functions import plot_dataset, \
    plot_diffCov_heatmap, plot_tSNE, plot_histogram, plot_and_save_diffused_fBm_snapshot


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
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print("Error {}; generating synthetic data\n".format(e))
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
            print("Error {}; training predictive LSTM\n".format(e))
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
            print("Error {}; training discriminative LSTM\n".format(e))
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
    df.to_csv(config.experiment_path, compression="gzip", index=True)
    print(pd.read_csv(config.experiment_path, compression="gzip", index_col=[0]))


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
    if config.plot:
        plot_diffCov_heatmap(true_cov, gen_cov, annot=config.annot_heatmap,
                             image_path=config.image_path + "_diffCov.png")
        plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter.png")
        # plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.plot,
        #                          timeDim=config.timeDim, image_path=config.image_path)
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
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print("Error {}; generating synthetic data\n".format(e))
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
            print("Error {}; training predictive LSTM\n".format(e))
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
            print("Error {}; training discriminative LSTM\n".format(e))
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
        exp_dict = evaluate_fBm_performance(true_samples, fBm_samples.cpu().numpy(), rng=rng, config=config,
                                            exp_dict=exp_dict)
        agg_dict[j] = exp_dict
    df = pd.DataFrame.from_dict(data=agg_dict)
    df.index = config.exp_keys
    df.to_csv(config.experiment_path, compression="gzip",
              index=True)
    print(pd.read_csv(config.experiment_path, compression="gzip", index_col=[0]))


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
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                                isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)
    exp_dict[config.exp_keys[1]] = 100 * np.mean(np.abs((gen_cov - expec_cov) / expec_cov))

    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of exact fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst, samples=reduce_to_fBn(true_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    exp_dict[config.exp_keys[2]] = c2[0]
    exp_dict[config.exp_keys[3]] = c2[2]
    exp_dict[config.exp_keys[4]] = c2[1]
    print(
        "Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], np.sum(c2[1]),
                                                                                                   c2[2]))
    # Chi-2 test for joint distribution of synthetic fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst,
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

    if config.plot:
        plot_diffCov_heatmap(expec_cov, gen_cov, annot=config.annot_heatmap,
                             image_path=config.image_path + "_diffCov.png")
        plot_tSNE(x=true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"],
                  image_path=config.image_path + "_tSNE.png") \
            if config.timeDim > 2 else plot_dataset(true_samples, generated_samples,
                                                    image_path=config.image_path + "_scatter.png")
        # plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.plot,
        #                          timeDim=config.timeDim, image_path=config.image_path)
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
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        try:
            data = np.load(config.data_path, allow_pickle=True)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print("Error {}; generating synthetic data\n".format(e))
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
    df.to_csv(config.experiment_path, compression="gzip", index=True)
    print(pd.read_csv(config.experiment_path, compression="gzip", index_col=[0]))


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

    if config.plot:
        plot_diffCov_heatmap(true_cov, gen_cov, annot=config.annot_heatmap,
                             image_path=config.image_path + "_diffCov.png")
        plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter.png")
        # plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.plot,
        #                          timeDim=config.timeDim, image_path=config.image_path)
    return exp_dict


def plot_fBm_results_from_csv(config: ConfigDict) -> None:
    """
    Function to plot quantitative metrics
        :param config: ML experiment metrics
        :return: None
    """
    df = pd.read_csv(config.experiment_path, compression="gzip", index_col=[0])

    # Mean Abs Difference
    # plot_and_save_boxplot(data=df.loc[config.exp_keys[0]].astype(float).to_numpy(), xlabel="1",
    #                      ylabel=config.exp_keys[0], title_plot="Mean Absolute Percentage Difference in Mean Vector",
    #                      dataLabels=[None], toSave=False, saveName="")

    # Covariance Abs Difference
    # plot_and_save_boxplot(data=df.loc[config.exp_keys[1]].astype(float).to_numpy(), xlabel="1",
    #                      ylabel=config.exp_keys[1], title_plot="Absolute Percentage Difference in Covariance Matrix",
    #                      dataLabels=[None], toSave=False, saveName="")

    # Exact Sample Chi2 Test Statistic Histogram
    dfs = config.timeDim
    fig, ax = plt.subplots()
    org_chi2 = df.loc[config.exp_keys[4]].to_list()
    true_chi2 = []
    for j in range(config.num_runs):
        true_chi2 += (ast.literal_eval(org_chi2[j]))
    xlinspace = np.linspace(scipy.stats.chi2.ppf(0.0001, dfs), scipy.stats.chi2.ppf(0.9999, dfs), 1000)
    pdfvals = scipy.stats.chi2.pdf(xlinspace, df=dfs)
    plot_histogram(np.array(true_chi2), pdf_vals=pdfvals, xlinspace=xlinspace, num_bins=200, xlabel="Chi2 Statistic",
                   ylabel="density", plotlabel="Chi2 with {} DoF".format(dfs),
                   plottitle="Histogram of exact samples' Chi2 Test Statistic", fig=fig, ax=ax)
    plt.show()
    print(ks_2samp(true_chi2, scipy.stats.chi2.rvs(df=dfs, size=len(true_chi2)), alternative="two-sided"))
    # Synthetic Sample Chi2 Test Statistic Histogram
    fig, ax = plt.subplots()
    f_chi2 = df.loc[config.exp_keys[5]].to_list()
    synth_chi2 = []
    for j in range(config.num_runs):
        synth_chi2 += (ast.literal_eval(f_chi2[j]))
    plot_histogram(np.array(synth_chi2), pdf_vals=pdfvals, xlinspace=xlinspace, num_bins=200, xlabel="Chi2 Statistic",
                   ylabel="density", plotlabel="Chi2 with {} DoF".format(dfs),
                   plottitle="Histogram of synthetic samples' Chi2 Test Statistic", fig=fig, ax=ax)
    plt.show()
    print(ks_2samp(synth_chi2, scipy.stats.chi2.rvs(df=dfs, size=len(synth_chi2)), alternative="two-sided"))

    """if str(df.loc[config.exp_keys[7]][0]) != "nan":
        # Predictive Scores
        org_pred = df.loc[config.exp_keys[7]].astype(float).to_numpy().reshape((config.num_runs,))
        synth_pred = df.loc[config.exp_keys[8]].astype(float).to_numpy().reshape((config.num_runs,))
        plot_and_save_boxplot(data=np.array([org_pred, synth_pred]).reshape((config.num_runs, 2)), xlabel="1",
                              ylabel=config.exp_keys[5],
                              title_plot="Predictive Scores", dataLabels=["True", "Generated"], toSave=False,
                              saveName="")
    if str(df.loc[config.exp_keys[9]][0]) != "nan":
        # Discriminative Scores
        org_disc = df.loc[config.exp_keys[9]].astype(float).to_numpy().reshape((config.num_runs,))
        synth_disc = df.loc[config.exp_keys[10]].astype(float).to_numpy().reshape((config.num_runs,))
        plot_and_save_boxplot(data=np.array([org_disc, synth_disc]).reshape((config.num_runs, 2)), xlabel="1",
                              ylabel=config.exp_keys[5],
                              title_plot="Discriminative Scores", dataLabels=["True", "Generated"], toSave=False,
                              saveName="")
    """
    # Histogram of exact samples Hurst parameter
    fig, ax = plt.subplots()
    ax.axvline(x=config.hurst, color="blue", label="True Hurst")

    literal_trues = df.loc[config.exp_keys[11]].to_list()
    true_Hs = []
    for j in range(config.num_runs):
        true_Hs += (ast.literal_eval(literal_trues[j]))
    plot_histogram(np.array(true_Hs), num_bins=200, xlabel="H", ylabel="density",
                   plottitle="Histogram of exact samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.show()

    # Histogram of exact samples Hurst parameter
    fig, ax = plt.subplots()
    ax.axvline(x=config.hurst, color="blue", label="True Hurst")
    literal_synths = df.loc[config.exp_keys[12]].to_list()
    synth_Hs = []
    for j in range(config.num_runs):
        synth_Hs += (ast.literal_eval(literal_synths[j]))
    plot_histogram(np.array(synth_Hs), num_bins=200, xlabel="H", ylabel="density",
                   plottitle="Histogram of synthetic samples' estimated Hurst parameter", fig=fig, ax=ax)
    plt.show()

    print(ks_2samp(synth_Hs, true_Hs, alternative="two-sided"))

    """pvals = df.loc[config.exp_keys[6]].to_list()
    for i in range(config.timeDim):
        pval = []
        for j in range(config.num_runs):
            pval_j = ast.literal_eval(pvals[j])
            pval.append(pval_j[i])
        plot_and_save_boxplot(data=np.array(pval), xlabel="1",
                              ylabel="KS Test p-value",
                              title_plot="KS p-val for dimension {}".format(i + 1), dataLabels=[None], toSave=False,
                              saveName="")"""


def run_fBm_VESDE_score_error_experiment(dataSize: int, diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion],
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
    except AssertionError:
        raise ValueError("Final time during sampling should be at least as large as final time during training")
    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    # Compute covariance function to compute exact score afterwards
    fBm_cov = torch.from_numpy(
        compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim, isUnitInterval=True)).to(
        torch.float32)

    # Placeholder
    errors = torch.zeros(size=(config.max_diff_steps, config.timeDim))

    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps)
    x = diffusion.prior_sampling(shape=(dataSize, config.timeDim)).to(device)  # Move to correct device
    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Score Error Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)

        # Obtain required diffusion parameters
        if config.predictor_model == "ancestral":
            pred_score, drift, diffusion_param = diffusion.get_ancestral_sampling(x, t=timesteps[i] * torch.ones(
                (x.shape[0], 1)).to(device), score_network=scoreModel,diff_index=diff_index, max_diff_steps=config.max_diff_steps)
        else:
            dt = -config.end_diff_time / config.max_diff_steps
            pred_score, drift, diffusion_param = diffusion.get_reverse_sde(x, score_network=scoreModel,
                                                                           t=timesteps[i] * torch.ones(
                                                                               (x.shape[0], 1)).to(device),
                                                                           dt=torch.Tensor([dt]).to(device))

        exp_score = torch.stack([-torch.linalg.inv((diffusion.get_ancestral_var(max_diff_steps=config.max_diff_steps,
                                                                                diff_index=config.max_diff_steps - 1 - diff_index)) * torch.eye(
            config.timeDim) + fBm_cov) @ x[j,:] for j in range(dataSize)])

        errors[config.max_diff_steps - 1 - i, :] = torch.linalg.norm(pred_score - exp_score, ord=2, axis=0)

        # One-step reverse-time SDE
        x = drift + diffusion_param * torch.randn_like(x)

    return errors


def run_fBm_perfect_VESDE_score(dataSize: int, dim_pair: torch.Tensor,
                                diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion], folderPath:str, gifPath:str,
                                rng: np.random.Generator, perfect_config: ConfigDict) -> None:
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
        assert(dim_pair.shape[0] == 2)
    except AssertionError:
        raise ValueError("You can only choose a pair of dimensions to plot\n")
    if perfect_config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    x = diffusion.prior_sampling(shape=(dataSize, dim_pair.shape[0])).to(device)  # Move to correct device

    fBm_cov = torch.from_numpy(
        compute_fBm_cov(FractionalBrownianNoise(H=perfect_config.hurst, rng=rng), T=perfect_config.timeDim, isUnitInterval=True)).to(
        torch.float32)
    fBm_cov = torch.index_select(torch.index_select(fBm_cov, dim=0, index=dim_pair), dim=1, index=dim_pair)

    for i in tqdm(iterable=(range(0, perfect_config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Backward Diffusion Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        max_diff_steps = torch.Tensor([perfect_config.max_diff_steps]).to(device)

        eff_time = diffusion.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=max_diff_steps - 1 - diff_index)
        cov = eff_time*torch.eye(dim_pair.shape[0]) + fBm_cov

        # Compute exact score
        exp_score = torch.stack([-torch.linalg.inv(cov) @ x[j,:] for j in range(dataSize)])
        if perfect_config.predictor_model == "ancestral":
            drift = diffusion.get_ancestral_drift(x=x, pred_score=exp_score, diff_index=diff_index,
                                                  max_diff_steps=max_diff_steps)
            diffusion_param = diffusion.get_ancestral_diff(diff_index=diff_index, max_diff_steps=max_diff_steps)
        else:
            raise ValueError("Alternative to ancestral sampling has not been implemented\n")
        if i % perfect_config.save_freq == 0 or i == (perfect_config.max_diff_steps - 1):
            save_path = folderPath + gifPath + "_diffIndex_{}.png".format(i + 1)
            xlabel = "fBm Dimension {}".format(dim_pair[0]+1)
            ylabel = "fBm Dimension {}".format(dim_pair[1]+1)
            plot_title = "Reverse-time samples $T={}$ at time {}".format(perfect_config.timeDim, round((
                                                                                                       i + 1) / perfect_config.max_diff_steps,
                                                                                           5))
            plot_and_save_diffused_fBm_snapshot(samples=x, cov=cov, save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

        x = drift + diffusion_param * torch.randn_like(x)


def run_fBm_VESDE_score(dataSize: int, dim_pair: torch.Tensor, scoreModel: Union[NaiveMLP, TimeSeriesScoreMatching],
                                diffusion: Union[OUSDEDiffusion, VPSDEDiffusion, VESDEDiffusion], folderPath:str, gifPath:str,
                                rng: np.random.Generator, config: ConfigDict) -> None:
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
    try:
        assert(dim_pair.shape[0] == 2)
    except AssertionError:
        raise ValueError("You can only choose a pair of dimensions to plot\n")
    if config.has_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    x = diffusion.prior_sampling(shape=(dataSize, dim_pair.shape[0])).to(device)  # Move to correct device

    fBm_cov = torch.from_numpy(
        compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim, isUnitInterval=True)).to(
        torch.float32)
    fBm_cov = torch.index_select(torch.index_select(fBm_cov, dim=0, index=dim_pair), dim=1, index=dim_pair)
    timesteps = torch.linspace(start=config.end_diff_time, end=config.sample_eps, steps=config.max_diff_steps)

    for i in tqdm(iterable=(range(0, config.max_diff_steps)), dynamic_ncols=False,
                  desc="Sampling for Backward Diffusion Visualisation :: ", position=0):
        diff_index = torch.Tensor([i]).to(device)
        max_diff_steps = torch.Tensor([config.max_diff_steps]).to(device)

        eff_time = diffusion.get_ancestral_var(max_diff_steps=max_diff_steps, diff_index=max_diff_steps - 1 - diff_index)
        cov = eff_time*torch.eye(dim_pair.shape[0]) + fBm_cov

        # Compute exact score
        if config.predictor_model == "ancestral":
            pred_score, drift, diffusion_param = diffusion.get_ancestral_sampling(x, t=timesteps[i] * torch.ones(
                (x.shape[0], 1)).to(device), score_network=scoreModel, diff_index=diff_index,
                                                                                  max_diff_steps=config.max_diff_steps)
        else:
            dt = -config.end_diff_time / config.max_diff_steps
            pred_score, drift, diffusion_param = diffusion.get_reverse_sde(x, score_network=scoreModel,
                                                                           t=timesteps[i] * torch.ones(
                                                                               (x.shape[0], 1)).to(device),
                                                                           dt=torch.Tensor([dt]).to(device))
        if i % config.save_freq == 0 or i == (config.max_diff_steps - 1):
            save_path = folderPath + gifPath + "_diffIndex_{}.png".format(i + 1)
            xlabel = "fBm Dimension {}".format(dim_pair[0]+1)
            ylabel = "fBm Dimension {}".format(dim_pair[1]+1)
            plot_title = "Reverse-time samples $T={}$ at time {}".format(config.timeDim, round((
                                                                                                       i + 1) / config.max_diff_steps,
                                                                                           5))
            plot_and_save_diffused_fBm_snapshot(samples=x, cov=cov, save_path=save_path, x_label=xlabel,
                                                y_label=ylabel, plot_title=plot_title)

        x = drift + diffusion_param * torch.randn_like(x)




