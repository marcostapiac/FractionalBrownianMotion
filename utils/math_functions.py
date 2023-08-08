import multiprocessing as mp
from types import NoneType
from typing import Union

import numpy as np
from numpy import broadcast_to, log, exp
from scipy.stats import chi2
from sklearn import datasets
from tqdm import tqdm

from src.classes import ClassFractionalBrownianNoise
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV


def logsumexp(w: np.ndarray, h: callable, x: np.ndarray, axis=0, isLog=False):
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T
    if isLog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)


def acf(data, size=None):
    # Nearest size with power of 2
    N = data.shape[0]
    if size is None:
        size = 2 ** np.ceil(np.log2(2 * N - 1)).astype('int')
    ndata = data - np.mean(data)
    fft = np.fft.fft(ndata, size)
    pwr = np.abs(fft) ** 2
    acorr = np.fft.ifft(pwr).real * np.power(np.var(data) * N, -1)
    return acorr


def calc_quantile_CRPS(target: np.ndarray, forecast: np.ndarray, mean_scaler: float = 0., scaler: float = 1.) -> float:
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    S, T = target.shape
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    CRPS = 0.
    for i in range(len(quantiles)):
        q_pred = []
        for k in range(S):
            for j in range(T):
                q_pred.append(np.quantile(forecast[k, j: j + 1], quantiles[i], axis=1))
            q_pred = np.concatenate(q_pred, 0)
            q_loss = 2 * np.sum(np.abs((q_pred - target) * ((target[k, j] <= q_pred) * 1.0 - quantiles[i])))
            CRPS += q_loss / denom
    return CRPS / len(quantiles)


def MMD_statistic(data: np.ndarray, n1: int, permute: bool = True) -> float:
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

    Parameters:
    data1 (numpy.ndarray): Data array for group 1 with shape (n1, T).
    data2 (numpy.ndarray): Data array for group 2 with shape (n2, T).
    num_permutations (int): Number of permutations to perform.

    Returns:
    float: The p-value of the permutation test.
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


def generate_fBn(H: float, T: int, S: int, rng: np.random.Generator) -> np.array:
    """ Always generate in unit time interval """
    generator = FractionalBrownianNoise(H=H, rng=rng)
    # pool = mp.Pool(mp.cpu_count())
    # data = pool.starmap(generator.circulant_simulation, tqdm([(T,None) for _ in range(S)]))
    # pool.close()
    data = np.zeros((S, T))
    for i in tqdm(range(S)):
        data[i, :] = generator.circulant_simulation(T, None)
    return np.array(data).reshape((S, T))


def generate_fBm(H: float, T: int, S: int, rng: np.random.Generator) -> np.array:
    data = generate_fBn(H=H, T=T, S=S, rng=rng)
    return np.cumsum(data, axis=1)


def generate_CEV(H: float, T: int, S: int, alpha: float, sigmaX: float, muU: float, muX: float, X0: float,
                 U0: float, rng: np.random.Generator) -> np.ndarray:
    cevGen = FractionalCEV(muU=muU, alpha=alpha, sigmaX=sigmaX, muX=muX, X0=X0, U0=U0, rng=rng)
    # pool = mp.Pool(mp.cpu_count())
    # data = pool.starmap(cevGen.state_simulation, tqdm([(H, T, 1./T) for _ in range(S)]))
    # pool.close()
    data = np.zeros((S, T))
    for i in tqdm(range(S)):
        data[i, :] = cevGen.state_simulation(H, T, 1. / T)[1:]
    return data  # np.array(data)[:, 1:].reshape((S, T))  # Remove initial value at t=0


def fBm_to_fBn(fBm_timeseries: np.ndarray) -> np.array:
    T = fBm_timeseries.shape[1]
    return fBm_timeseries - np.insert(fBm_timeseries[:, :T - 1], 0, 0., axis=1)


def compute_fBn_cov(fBn_generator: ClassFractionalBrownianNoise, td: int, isUnitInterval: bool) -> np.ndarray:
    # td is the dimensionality of the time-series, but we enforce natural time in [0,1.]
    cov = (np.atleast_2d(
        [[fBn_generator.covariance((i - j)) for j in range(td)] for i in
         tqdm(range(td))]))
    if isUnitInterval: cov *= np.power(1. / td, 2. * fBn_generator.H)
    return cov


def compute_fBm_cov(fBn_generator: ClassFractionalBrownianNoise, td: int, isUnitInterval: bool) -> np.ndarray:
    # td is the dimensionality of the time-series, but we enforce natural time in [0,1.]
    cov = (np.atleast_2d(
        [[fBn_generator.fBm_covariance(i, j) for j in range(1, td + 1)] for i
         in
         tqdm(range(1, td + 1))]))
    if isUnitInterval: cov *= np.power(1. / td, 2. * fBn_generator.H)
    return cov


def chiSquared_test(T: int, H: float, isUnitInterval: bool, samples: Union[np.ndarray, NoneType] = None, M: Union[int, NoneType] = None,
                    invL: Union[np.ndarray, NoneType] = None) -> [float, float, float]:
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


def generate_circles(T: int, S: int) -> np.array:
    assert (T == 2)
    X, y = datasets.make_circles(
        n_samples=S, noise=0.0, random_state=None, factor=.5)
    sample = X * 4
    return sample
