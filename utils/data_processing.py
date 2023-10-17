import numpy as np
import pandas as pd
from ml_collections import ConfigDict
from torch.distributed import destroy_process_group

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV
from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTM import DiscriminativeLSTM
from src.evaluation_pipeline.classes.PredictiveLSTM.ClassPredictiveLSTM import PredictiveLSTM
from src.evaluation_pipeline.data_processing import test_predLSTM, test_discLSTM
from utils.math_functions import chiSquared_test, reduce_to_fBn, compute_fBm_cov, permutation_test, \
    energy_statistic, MMD_statistic
from utils.plotting_functions import plot_final_diff_marginals, plot_dataset, \
    plot_diffCov_heatmap, \
    plot_subplots


def evaluate_fBm_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator,
                             config: ConfigDict, exp_dict:dict) -> dict:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples of fractional Brownian motion (or its increments)
        :param generated_samples: Final reverse-time diffusion samples
        :param h: Hurst index
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
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                                isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)
    exp_dict[config.exp_keys[1]] = np.mean(np.abs(gen_cov-expec_cov)/expec_cov)


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
    ps = plot_final_diff_marginals(true_samples, generated_samples, print_marginals=config.print_marginals, timeDim=config.timeDim, image_path=config.image_path)
    exp_dict[config.exp_keys[6]] = ps

    if config.test_lstm:
        # Predictive LSTM test
        org, synth = test_predLSTM(original_data=true_samples, synthetic_data=generated_samples, model=PredictiveLSTM(ts_dim=1),
                      config=config)
        exp_dict[config.exp_keys[7]] = org
        exp_dict[config.exp_keys[8]] = synth
        destroy_process_group()

        # Discriminative LSTM test
        org, synth = test_discLSTM(original_data=true_samples, synthetic_data=generated_samples,
                                   model=DiscriminativeLSTM(ts_dim=1),
                                   config=config)
        exp_dict[config.exp_keys[9]] = org
        exp_dict[config.exp_keys[10]] = synth
        destroy_process_group()

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


def compute_circle_proportions(true_samples: np.ndarray, generated_samples: np.ndarray) -> float:
    """
    Function computes approximate ratio of samples in inner vs outer circle of circle dataset
        :param true_samples: data samples exactly using sklearn "make_circle" function
        :param generated_samples: final reverse-time diffusion samples
        :return: Ratio of circle proportions
    """
    innerb = 0
    outerb = 0
    innerf = 0
    outerf = 0
    S = true_samples.shape[0]
    for i in range(S):
        bkwd = generated_samples[i]
        fwd = true_samples[i]
        rb = np.sqrt(bkwd[0] ** 2 + bkwd[1] ** 2)
        rf = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
        if rb <= 2.1:
            innerb += 1
        elif 3.9 <= rb:
            outerb += 1
        if rf <= 2.1:
            innerf += 1
        elif 3.9 <= rf:
            outerf += 1

    print("Generated: Inner {} vs Outer {}".format(innerb / S, outerb / S))
    print("True: Inner {} vs Outer {}".format(innerf / S, outerf / S))
    return innerf/innerb


def evaluate_circle_performance(true_samples: np.ndarray, generated_samples: np.ndarray, config: ConfigDict, exp_dict:dict) -> dict:
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
    exp_dict[config.exp_keys[0]] = np.mean(np.abs(gen_mean-true_mean)/true_mean)

    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)
    exp_dict[config.exp_keys[1]] = np.mean(np.abs(gen_cov-true_cov)/true_cov)

    plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter.png")
    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=gen_cov, image_path=config.image_path + "_scatter.png")
    ps = plot_final_diff_marginals(true_samples, generated_samples, timeDim=2, print_marginals=config.print_marginals, image_path=config.image_path)
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


def gen_and_store_statespace_data(Xs=None, Us=None, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11,
                                  T=1e-3 * 2 ** 11)->None:
    """
    Generate observation and latent signal from CEV model and store it as pickle file
        :param Xs: Optional parameter, array containing latent signal process
        :param Us: Optional parameter, array containing observation process
        :param muU: Drift parameter for observation process
        :param muX: Drift parameter for latent process
        :param gamma: Mean reversion parameter for latent process
        :param X0: Initial value for latent process
        :param U0: Initial value for observation process
        :param H: Hurst Index
        :param N: Length of time series
        :param T: Terminal simualtion time for a process on [0, T]
        :return: None
    """
    assert ((Xs is None and Us is None) or (Xs is not None and Us is not None))
    sigmaX = np.sqrt(muX * gamma / 0.55)
    alpha = gamma / sigmaX
    deltaT = T / N
    if Xs is None:
        m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, alpha=alpha, X0=X0, U0=U0)
        Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    df = pd.DataFrame.from_dict(data={'Log-Price': Us, 'Volatility': Xs})
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), np.array([Xs, Us]), np.array([None, None]),
                  np.array(["Time", "Time"]),
                  np.array(["Volatility", "Log Price"]),
                  "Project Model Simulation")
    df.to_csv('../data/raw_data_simpleObsModel_{}_{}.csv'.format(int(np.log2(N)), int(10 * H)), index=False)

def energy_csv_to_df() -> pd.DataFrame:
    """
    Turn energy data from https://github.com/jsyoon0823/TimeGAN/blob/master/data/stock_data.csv to Pandas Df
        :return: Dataframe
    """
    df = pd.read_csv(project_config.ROOT_DIR + "data/energy_data.csv")
    print(df.columns)

def stock_csv_to_df()->pd.DataFrame:
    """
    Turn stock data from https://github.com/jsyoon0823/TimeGAN/blob/master/data/stock_data.csv to Pandas Df
        :return: Dataframe
    """
    df = pd.read_csv(project_config.ROOT_DIR + "data/stock_data.csv")
    df.index.name = "GOOGLE"
    print(df)

energy_csv_to_df()
stock_csv_to_df()
