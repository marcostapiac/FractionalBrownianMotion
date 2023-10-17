import multiprocessing as mp
from types import NoneType
from typing import Union

import numpy as np
from numpy import broadcast_to, log, exp
from scipy.stats import chi2
from tqdm import tqdm

from src.classes import ClassFractionalBrownianNoise
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV


def logsumexp(w: np.ndarray, x: np.ndarray, h: callable, axis: int = 0, isLog: bool = False):
    """
    Function to efficiently compute weighted mean of transformed data
        :param w: Logarithmic weights
        :param x: Data to compute weighted mean
        :param h: Function to transform data
        :param axis: Indicates along which axis to compute summation
        :param isLog: Indicates whether to return log (True) or exact probabilities
        :return: Weighted mean of data
    """
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T  # Broadcast "w-c" into shape of data
    if isLog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))  # Compute element-wise multiplication and sum
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)


def acf(data: np.ndarray, size: Union[NoneType, int] = None):
    """
    Function to compute auto-correlation function
        :param data: Dataset
        :param size: Number of FFT points
        :return:  Auto-correlation function evaluated on N discrete points
    """
    # Nearest size with power of 2
    N = data.shape[0]
    if size is None:
        size = 2 ** np.ceil(np.log2(2 * N - 1)).astype('int')
    ndata = data - np.mean(data)
    fft = np.fft.fft(ndata, size)
    pwr = np.abs(fft) ** 2
    acorr = np.fft.ifft(pwr).real * np.power(np.var(data) * N, -1)
    return acorr


def MMD_statistic(data: np.ndarray, n1: int, permute: bool = True) -> float:
    """
    Function to cmpute Maximum Mean Discrepancy
        :param data: Data on which to compute statistic
        :param n1: Number of datapoints to consider
        :param permute: Indicates whether to permute data (True) or not
        :return: MMD statistic
    """
    if permute: np.random.shuffle(data)
    x, y = data[:n1, :], data[n1:, :]

    assert (x.shape == y.shape and len(x.shape) == 2)
    S, T = x.shape
    bx = np.stack([x] * S, axis=1)
    bbx = np.transpose(np.stack([x] * S, axis=1), axes=(1, 0, 2))
    by = np.stack([y] * S, axis=1)
    bby = np.transpose(np.stack([y] * S, axis=1), axes=(1, 0, 2))
    XX = np.power(np.linalg.norm(bx - bbx, axis=2, ord=2), 2)
    YY = np.power(np.linalg.norm(by - bby, axis=2, ord=2), 2)
    XY = np.power(np.linalg.norm(bx - bby, axis=2, ord=2), 2)
    bandwidth_range = [10., 15., 20., 50.]
    eXX, eYY, eXY = np.zeros(shape=(S, S)), np.zeros(shape=(S, S)), np.zeros(shape=(S, S))
    for a in bandwidth_range:
        eXX += (np.exp(-0.5 * np.power(a, -1) * XX))
        eYY += np.exp(-0.5 * np.power(a, -1) * YY)
        eXY += np.exp(-0.5 * np.power(a, -1) * XY)
    return np.sum((eXX - 2. * eXY + eYY)) / (S * S)


def energy_statistic(data: np.ndarray, n1: int, permute: bool = True) -> float:
    """
    Function to compute energy statistic
        :param data: Data on which to compute statistic
        :param n1: Number of datapoints to consider
        :param permute: Indicates whether to permute data (True) or not
        :return: Energy statistic
    """
    if permute: np.random.shuffle(data)
    x, y = data[:n1, :], data[n1:, :]
    S, T = x.shape
    bx = np.stack([x] * S, axis=1)
    bbx = np.transpose(np.stack([x] * S, axis=1), axes=(1, 0, 2))
    by = np.stack([y] * S, axis=1)
    bby = np.transpose(np.stack([y] * S, axis=1), axes=(1, 0, 2))
    eXX = (np.linalg.norm(bx - bbx, axis=2, ord=2))
    eYY = (np.linalg.norm(by - bby, axis=2, ord=2))
    eXY = (np.linalg.norm(bx - bby, axis=2, ord=2))
    return float(np.mean(2. * eXY - eXX - eYY))


def permutation_test(data1: np.ndarray, data2: np.ndarray, num_permutations: int, compute_statistic: callable) -> float:
    """
    Perform a permutation test for samples from multivariate distributions.
        :param data1: (numpy.ndarray): Data array for group 1 with shape (n1, T)
        :param data2: (numpy.ndarray): Data array for group 2 with shape (n2, T)
        :param num_permutations: (int): Number of permutations to perform
        :param compute_statistic: function to compute statistic
        :return: The p-value of the permutation test.
    """
    combined_data = np.concatenate((data1, data2), axis=0)
    assert (data1.shape[0] == data2.shape[0] and data1.shape[1] == data2.shape[1])
    assert (combined_data.shape[0] == int(2 * data1.shape[0]) and combined_data.shape[1] == data1.shape[1])
    n1 = data1.shape[0]
    observed_statistic = compute_statistic(combined_data, n1, permute=False)

    pool = mp.Pool(mp.cpu_count())
    permuted_statistics = pool.starmap(compute_statistic, tqdm([(combined_data, n1) for _ in range(num_permutations)]))
    pool.close()

    larger_count = np.sum(1. * (np.abs((permuted_statistics)) >= np.abs(observed_statistic)))
    p_value = (larger_count + 1) / (num_permutations + 1)
    return p_value


def generate_fBn(H: float, T: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """
    Function generates samples of fractional Brownian noise
        :param H: (float) Hurst parameter
        :param T: (int) Length of each samples
        :param S: (int) Number of samples
        :param rng: (random.Generator) Default random number generator
        :return: (np.ndarray) fBn samples
    """
    generator = FractionalBrownianNoise(H=H, rng=rng)
    data = np.zeros((S, T))
    for i in tqdm(range(S)):
        data[i, :] = generator.circulant_simulation(T, None)
    return np.array(data).reshape((S, T))


def generate_fBm(H: float, T: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """
    Function generates samples of fractional Brownian motion
        :param H: Hurst parameter
        :param T: Length of each sample
        :param S: Number of samples
        :param rng: Random number generator
        :return: fBm samples
    """
    data = generate_fBn(H=H, T=T, S=S, rng=rng)
    return np.cumsum(data, axis=1)


def generate_CEV(H: float, T: int, S: int, alpha: float, sigmaX: float, muU: float, muX: float, X0: float,
                 U0: float, rng: np.random.Generator) -> np.ndarray:
    """
    Function generates samples of latent signal from Constant Elasticity of Variance model
        :param H: Hurst index
        :param T: Length of time series
        :param S: Number of samples
        :param alpha: Mean reversion parameter in latent process
        :param sigmaX: Volatility parameter in latent signal
        :param muU: Drift parameter in observation process
        :param muX: Drift parameter in latent process
        :param X0: Initial value for latent process
        :param U0: Initial value for observation process
        :param rng: Random number generator
        :return: CEV samples
    """
    cevGen = FractionalCEV(muU=muU, alpha=alpha, sigmaX=sigmaX, muX=muX, X0=X0, U0=U0, rng=rng)
    data = np.zeros((S, T))
    for i in tqdm(range(S)):
        data[i, :] = cevGen.state_simulation(H, T, 1. / T)[1:]  # Remove initial value at t=0
    return data


def reduce_to_fBn(timeseries: np.ndarray, reduce: bool) -> np.ndarray:
    """
    Tranform samples of
        :param timeseries: fBm/fBn samples
        :param reduce: Indicates whether timeseries is fBm (True) or fBn (false)
        :return: fBn samples
    """
    T = timeseries.shape[1]
    if reduce:
        return timeseries - np.insert(timeseries[:, :T - 1], 0, 0., axis=1)
    return timeseries


def compute_fBn_cov(fBn_generator: ClassFractionalBrownianNoise, T: int, isUnitInterval: bool) -> np.ndarray:
    """
    Compute covariance matrix of Fractional Brownian Noise
    :param fBn_generator: Class defining process generator
    :param T: Covariance matrix dimensionality
    :param isUnitInterval: Indicates whether to re-scale fBn covariance from [0,td] to [0,1]
    :return: Covariance matrix
    """
    cov = (np.atleast_2d(
        [[fBn_generator.covariance((i - j)) for j in range(T)] for i in
         tqdm(range(T))]))
    if isUnitInterval: cov *= np.power(1. / T, 2. * fBn_generator.H)
    return cov


def compute_fBm_cov(fBn_generator: ClassFractionalBrownianNoise, T: int, isUnitInterval: bool) -> np.ndarray:
    """
        Compute covariance matrix of Fractional Brownian Noise
        :param fBn_generator: Class defining process generator
        :param T: Covariance matrix dimensionality
        :param isUnitInterval: Indicates whether to re-scale fBn covariance from [0,td] to [0,1]
        :return: Covariance matrix
    """
    cov = (np.atleast_2d(
        [[fBn_generator.fBm_covariance(i, j) for j in range(1, T + 1)] for i
         in
         tqdm(range(1, T + 1))]))
    if isUnitInterval: cov *= np.power(1. / T, 2. * fBn_generator.H)
    return cov


def chiSquared_test(T: int, H: float, isUnitInterval: bool, samples: Union[np.ndarray, NoneType] = None,
                    M: Union[int, NoneType] = None,
                    invL: Union[np.ndarray, NoneType] = None) -> [float, float, float]:
    """
    Function which compute chi-squared test from Ton Dieker's 2004 thesis see
    http://www.columbia.edu/~ad3217/fbm/thesisold.pdf
        :param T: Length of each sample
        :param H: Hurst index
        :param isUnitInterval: Indicates whether fBn is generated on [0,1] (True) or [0, T]
        :param samples: fBn data
        :param M: Optional parameter which provides number of samples to consider for the test
        :param invL: Optional parameter which provides pre-computed inverse covariance matrix
        :return: Lower critical test value, critical statistic, Upper critical value
    """

    assert ((M is None and samples is not None) or (M is not None and samples is None))

    def standardise_sample(fBn_sample, invL):
        return np.squeeze(invL @ fBn_sample)

    def chiSquared(fBn_sample, invL):
        standard_sample = standardise_sample(fBn_sample, invL)
        return np.sum([i ** 2 for i in standard_sample])

    fBn = FractionalBrownianNoise(H)
    invL = invL if (invL is not None) else np.linalg.inv(np.linalg.cholesky(compute_fBn_cov(fBn, T, isUnitInterval)))
    alpha = 0.05
    S = samples.shape[0] if samples is not None else M
    critUpp = chi2.ppf(q=1. - 0.5 * alpha, df=S * T - 1)  # Upper alpha quantile, and dOf = T - 1
    critLow = chi2.ppf(q=0.5 * alpha, df=S * T - 1)  # Lower alpha quantile, and d0f = T -1
    ts = []
    for i in tqdm(range(S)):
        tss = chiSquared(samples[i, :], invL) if samples is not None else chiSquared(fBn.circulant_simulation(T), invL)
        ts.append(tss)
    return critLow, np.sum(ts), critUpp


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
    return innerf / innerb
