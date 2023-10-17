import numpy as np
import pandas as pd
from ml_collections import ConfigDict
from sklearn import datasets
from torch.distributed import init_process_group, destroy_process_group

from configs import project_config
from src.classes.ClassFractionalCEV import FractionalCEV
from utils.plotting_functions import plot_subplots


def gen_and_store_statespace_data(Xs=None, Us=None, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 11,
                                  T=1e-3 * 2 ** 11) -> None:
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
    return df


def stock_csv_to_df() -> pd.DataFrame:
    """
    Turn stock data from https://github.com/jsyoon0823/TimeGAN/blob/master/data/stock_data.csv to Pandas Df
        :return: Dataframe
    """
    df = pd.read_csv(project_config.ROOT_DIR + "data/stock_data.csv")
    df.index.name = "GOOGLE"
    print(df)
    return df


def generate_circles(S: int, noise: float) -> np.ndarray:
    """
    Generate circle dataset
        :param S: Number of samples
        :param noise: Noise standard deviation on each sample
        :return: Circle samples
    """
    X, y = datasets.make_circles(
        n_samples=S, noise=noise, random_state=None, factor=.5)
    sample = X * 4
    return sample


def generate_sine_dataset(S: int, T: int) -> np.ndarray:
    """
    Generate uni-dimensional sine waves
        :param S: Number of time-series
        :param T: Length of time-series
        :return: Data
    """
    eta = np.random.uniform(low=0., high=1., size=S)
    theta = np.random.uniform(low=-np.pi, high=np.pi, size=S)
    kappa = np.random.normal(loc=10., size=S)
    time_space = np.linspace(0., 10., num=T)
    data = np.array(
        [kappa[i] * time_space + np.sin(2. * np.pi * eta[i] * time_space + theta[i]) for i in range(S)]).reshape((S, T))
    return data


def ddp_setup(backend: str) -> None:
    """
    DDP setup to allow processes to discover and communicate with each other with TorchRun
    :param backend: Gloo vs NCCL for CPU vs GPU, respectively
    :return: None
    """
    init_process_group(backend=backend)


def init_experiments(config: ConfigDict) -> None:
    """
    Initiate DDP group process only once per experiment
        :param config: ML experiment configuration
        :return: None
    """
    if config.has_cuda:
        ddp_setup(backend="nccl")
    else:
        ddp_setup(backend="gloo")


def cleanup_experiments() -> None:
    try:
        destroy_process_group()
    except AssertionError as e:
        print("No process group to destroy\n")
