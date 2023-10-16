import numpy as np
import pandas as pd
from ml_collections import ConfigDict

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
    plot_tSNE, plot_subplots


def evaluate_fBm_performance(true_samples: np.ndarray, generated_samples: np.ndarray, rng: np.random.Generator,
                             config: ConfigDict) -> None:
    """
    Computes metrics to quantify how close the generated samples are from the desired distribution
        :param true_samples: Exact samples of fractional Brownian motion (or its increments)
        :param generated_samples: Final reverse-time diffusion samples
        :param h: Hurst index
        :param rng: Default random number generator
        :param config: Configuration dictionary for experiment
        :return: None
    """
    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data Covariance :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data Covariance :: ", gen_cov)
    expec_cov = compute_fBm_cov(FractionalBrownianNoise(H=config.hurst, rng=rng), T=config.timeDim,
                                isUnitInterval=config.unitInterval)
    print("Expected Covariance :: ", expec_cov)

    print(config.image_path)
    plot_diffCov_heatmap(expec_cov, gen_cov, annot=config.annot, image_path=config.image_path + "_diffCov")
    S = min(true_samples.shape[0], generated_samples.shape[0])
    true_samples, generated_samples = true_samples[:S], generated_samples[:S]

    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst, samples=reduce_to_fBn(true_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    print("Chi-Squared test for true: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                     c2[2]))
    # Chi-2 test for joint distribution of the fractional Brownian noise
    c2 = chiSquared_test(T=config.timeDim, H=config.hurst,
                         samples=reduce_to_fBn(generated_samples, reduce=config.isfBm),
                         isUnitInterval=config.unitInterval)
    print("Chi-Squared test for target: Lower Critical {} :: Statistic {} :: Upper Critical {}".format(c2[0], c2[1],
                                                                                                       c2[2]))

    plot_tSNE(x=true_samples, y=generated_samples, labels=["True Samples", "Generated Samples"],
              image_path=config.image_path + "_tSNE") \
        if config.timeDim > 2 else plot_dataset(true_samples, generated_samples,
                                                image_path=config.image_path + "_scatter")
    if config.eval_marginals:
        plot_final_diff_marginals(true_samples, generated_samples, timeDim=config.timeDim, image_path=config.image_path)

    if config.test_lstm:
        # Predictive LSTM test
        test_predLSTM(original_data=true_samples, synthetic_data=generated_samples, model=PredictiveLSTM(ts_dim=1),
                      config=config)

        # Discriminative LSTM test
        test_discLSTM(original_data=true_samples, synthetic_data=generated_samples, model=DiscriminativeLSTM(ts_dim=1),
                      config=config)

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


def compute_circle_proportions(true_samples: np.ndarray, generated_samples: np.ndarray) -> None:
    """
    Function computes approximate ratio of samples in inner vs outer circle of circle dataset
        :param true_samples: data samples exactly using sklearn's "make_circle" function
        :param generated_samples: final reverse-time diffusion samples
        :return: None
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


def evaluate_circle_performance(true_samples: np.ndarray, generated_samples: np.ndarray, config: ConfigDict) -> None:
    """
    Compute various quantitative and qualitative metrics on final reverse-time diffusion samples for circle dataset
        :param true_samples: Exact samples from circle dataset
        :param generated_samples: Final reverse-time diffusion samples
        :param config: Configuration dictionary for experiment
        :return: None
    """

    print("True Data Sample Mean :: ", np.mean(true_samples, axis=0))
    print("Generated Data Sample Mean :: ", np.mean(generated_samples, axis=0))
    true_cov = np.cov(true_samples, rowvar=False)
    print("True Data :: ", true_cov)
    gen_cov = np.cov(generated_samples, rowvar=False)
    print("Generated Data :: ", gen_cov)

    plot_dataset(true_samples, generated_samples, image_path=config.image_path + "_scatter")

    plot_diffCov_heatmap(true_cov=true_cov, gen_cov=gen_cov, image_path=config.image_path + "_scatter")
    plot_final_diff_marginals(true_samples, generated_samples, timeDim=2, image_path=config.image_path)

    compute_circle_proportions(true_samples, generated_samples)

    # Permutation test for kernel statistic
    test_L = min(2000, true_samples.shape[0])
    print("MMD Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=MMD_statistic,
                         num_permutations=1000)))
    # Permutation test for energy statistic
    print("Energy Permutation test: p-value {}".format(
        permutation_test(true_samples[:test_L], generated_samples[:test_L], compute_statistic=energy_statistic,
                         num_permutations=1000)))


def compare_fBm_to_approximate_fBm(generated_samples: np.ndarray, h: float, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing final reverse-time diffusion fBm samples to fBm samples generated from approximate simulation
    methods.
        :param generated_samples: Exact fBm samples
        :param h: Hurst index
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    generator = FractionalBrownianNoise(H=h, rng=rng)
    S = min(20000, generated_samples.shape[0])
    approx_samples = np.empty((S, td))
    for _ in range(S):
        approx_samples[_, :] = generator.paxon_simulation(
            N_samples=td).cumsum()  # TODO: Are we including initial sample?
    plot_tSNE(generated_samples, y=approx_samples,
              labels=["Reverse Diffusion Samples", "Approximate Samples: Paxon Method"],
              image_path=project_config.ROOT_DIR + "pngs/tSNE_approxfBm_vs_generatedfBm_H{:.3e}_T{}".format(h, td))


def compare_fBm_to_normal(h: float, generated_samples: np.ndarray, td: int, rng: np.random.Generator) -> None:
    """
    Plot tSNE comparing reverse-time diffusion samples to standard normal samples
        :param h: Hurst index.
        :param generated_samples: Exact fBm samples
        :param td: Dimension of each sample
        :param rng: Random number generator
        :return: None
    """
    S = min(20000, generated_samples.shape[0])
    normal_rvs = np.empty((S, td))
    for _ in range(S):
        normal_rvs[_, :] = rng.standard_normal(td)
    plot_tSNE(generated_samples, y=normal_rvs, labels=["Reverse Diffusion Samples", "Standard Normal RVS"],
              image_path=project_config.ROOT_DIR + "pngs/tSNE_normal_vs_generatedfBm_H{:.3e}_T{}".format(h, td))


def gen_and_store_statespace_data(Xs=None, Us=None, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11,
                                  T=1e-3 * 2 ** 11):
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
