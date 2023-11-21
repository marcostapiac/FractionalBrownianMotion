import numpy as np
import pandas as pd
import torch
import os
from ml_collections import ConfigDict
from sklearn import datasets
from torch.distributed import init_process_group, destroy_process_group

from configs import project_config

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


def generate_sine_dataset(S: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate uni-dimensional sine waves
        :param S: Number of time-series
        :param T: Length of time-series
        :param rng: Default random number generator
        :return: Data
    """
    eta = rng.uniform(low=0., high=1., size=S)
    theta = rng.uniform(low=-np.pi, high=np.pi, size=S)
    kappa = rng.normal(loc=10., size=S)
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


def init_experiment(config: ConfigDict) -> None:
    """
    Initiate DDP group process only once per experiment
        :param config: ML experiment configuration
        :return: None
    """
    if config.has_cuda:
        if int(os.environ["WORLD_SIZE"]) > torch.cuda.device_count():
            os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())
        ddp_setup(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        ddp_setup(backend="gloo")


def cleanup_experiment() -> None:
    try:
        destroy_process_group()
    except AssertionError as e:
        print("No process group to destroy\n")
