import multiprocessing as mp
from functools import partial
from math import gamma
from typing import Union, Tuple

import numpy as np
import roughpy as rhpy
import scipy.optimize as so
import torch
from ml_collections import ConfigDict
from numpy import broadcast_to, log, exp
from scipy.stats import chi2, kstest
from tqdm import tqdm

from configs.project_config import NoneType
from src.classes import ClassFractionalBrownianNoise
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from src.classes.ClassFractionalCEV import FractionalCEV
from src.classes.ClassFractionalOU import FractionalOU
from src.classes.ClassFractionalSin import FractionalSin


# from src.classes.ClassFractionalSin import FractionalSin


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

    larger_count = np.sum(1. * (np.abs(permuted_statistics) >= np.abs(observed_statistic)))
    p_value = (larger_count + 1) / (num_permutations + 1)
    return p_value


def parallel_fBn_generation(T: int, H: int, scaleUnitInterval: bool,
                            gaussRvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
    """
    Function which generates new random seed for rng to allow for parallel generation of fBN
        :param H: (float) Hurst parameter
        :param T: (int) Length of each samples
        :param S: (int) Number of samples
        :param gaussRvs: Pre-computed Gaussian random variables
        :param scaleUnitInterval: Whether to scale to unit time interval.
        :return: fBn sample
    """
    rng = np.random.default_rng(seed=np.random.seed())
    generator = FractionalBrownianNoise(H=H, rng=rng)
    return generator.circulant_simulation(N_samples=T, scaleUnitInterval=scaleUnitInterval, gaussRvs=gaussRvs)


def generate_fBn(H: float, T: int, S: int, isUnitInterval: bool,
                 rvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
    """
    Function generates samples of fractional Brownian noise
        :param H: (float) Hurst parameter
        :param T: (int) Length of each samples
        :param S: (int) Number of samples
        :param rvs: Pre-computed Gaussian random variables
        :param isUnitInterval: Whether to scale to unit time interval.
        :return: (np.ndarray) fBn samples
    """
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.starmap(
            partial(parallel_fBn_generation, H=H, scaleUnitInterval=isUnitInterval, gaussRvs=rvs),
            [(T,) for _ in range(S)])
    return np.array(result).reshape((S, T))


def generate_fBm(H: float, T: int, S: int, isUnitInterval: bool,
                 rvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
    """
    Function generates samples of fractional Brownian motion
        :param H: Hurst parameter
        :param T: Length of each sample
        :param S: Number of samples
        :param rvs: Pre-computed Gaussian random variables
        :param isUnitInterval: Whether to scale samples to unit time interval.
        :return: fBm samples
    """
    data = generate_fBn(H=H, T=T, S=S, rvs=rvs, isUnitInterval=isUnitInterval)
    return np.cumsum(data, axis=1)


def generate_fOU(H: float, T: int, S: int, isUnitInterval: bool, mean_rev: float, mean: float, diff: float,
                 initial_state: float,
                 rvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
    """
    Function generates samples of fractional Brownian motion
        :param H: Hurst parameter
        :param T: Length of each sample
        :param S: Number of samples
        :param rvs: Pre-computed Gaussian random variables
        :param isUnitInterval: Whether to scale samples to unit time interval.
        :return: fBm samples
    """
    deltaT = 1. / T if isUnitInterval else 1.
    fOU = FractionalOU(mean_rev=mean_rev, mean=mean, diff=diff, X0=initial_state)
    data = np.array(
        [fOU.euler_simulation(H=H, N=T, deltaT=deltaT, isUnitInterval=isUnitInterval, X0=None, Ms=None, gaussRvs=rvs)
         for _ in range(S)]).reshape(
        (S, T))
    assert (data.shape == (S, T))
    return data


def generate_fSin(H: float, T: int, S: int, isUnitInterval: bool, mean_rev: float, diff: float,
                  initial_state: float,
                  rvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
    """
    Function generates samples of fractional Brownian motion
        :param H: Hurst parameter
        :param T: Length of each sample
        :param S: Number of samples
        :param rvs: Pre-computed Gaussian random variables
        :param isUnitInterval: Whether to scale samples to unit time interval.
        :return: fBm samples
    """
    if isUnitInterval:
        deltaT = 1. / T
        t0 = 0.
        t1 = 1.
    else:
        deltaT = 1.
        t0 = 0.
        t1 = T
    fSin = FractionalSin(mean_rev=mean_rev, diff=diff, X0=initial_state)
    data = np.array(
        [fSin.euler_simulation(H=H, N=T, deltaT=deltaT, isUnitInterval=isUnitInterval, X0=None, Ms=None, gaussRvs=rvs,
                               t0=t0, t1=t1) for _ in range(S)]).reshape(
        (S, T))
    assert (data.shape == (S, T))
    return data


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
    # noinspection HttpUrlsUsage
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
    critUpp = chi2.ppf(q=1. - 0.5 * alpha, df=T - 1)  # Upper alpha quantile, and dOf = T - 1
    critLow = chi2.ppf(q=0.5 * alpha, df=T - 1)  # Lower alpha quantile, and d0f = T -1
    ts = []
    for i in tqdm(range(S)):
        tss = chiSquared(samples[i, :], invL) if samples is not None else chiSquared(
            fBn.circulant_simulation(T, scaleUnitInterval=isUnitInterval), invL)
        ts.append(tss)
    return critLow, ts, critUpp


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


def fBn_spectral_density(hurst: float, N: int) -> np.ndarray:
    """
    Spectral density of fBn
        :param hurst: Hurst parameter
        :param N: Number of observations of sample path
        :return: Spectral density
    """
    hhest = - ((2 * hurst) + 1)
    const = np.sin(np.pi * hurst) * gamma(-hhest) / np.pi  # TODO: Why dividing by np.pi?
    halfN = int((N - 1) / 2)
    dpl = 2 * np.pi * np.arange(1, halfN + 1) / N  # 2pi/N * [1, 2, 3, ..., N//2] (i.e., the frequencies)
    fspec = np.ones(halfN)
    for i in np.arange(0, halfN):
        dpfi = np.arange(0, 200)  # Start computation of B(i, H)
        dpfi = 2 * np.pi * dpfi  # 2*pi*freq
        fgi = (np.abs(dpl[i] + dpfi)) ** hhest  # TODO: Why np.abs()?
        fhi = (np.abs(dpl[i] - dpfi)) ** hhest
        dpfi = fgi + fhi
        dpfi[0] = dpfi[0] / 2
        dpfi = (1. - np.cos(dpl[i])) * const * dpfi
        fspec[i] = np.sum(dpfi)
    fspec = fspec / np.exp(2 * np.sum(np.log(fspec)) / N)  # fspec / (prod(fspec)**(2/N))
    return fspec


def whittle_ll(hurst: float, gammah: np.ndarray, nbpoints: int) -> float:
    """
    Function computes the Whittle likelihood
        :param hurst: Hurst index
        :param gammah: Function of the data
        :param nbpoints: Number of observation of path
        :return: Whittle likelihood
    """
    return 2. * (2. * np.pi / nbpoints) * np.sum((gammah / fBn_spectral_density(hurst, nbpoints)))


def optimise_whittle(idx: int, data: np.ndarray) -> Tuple[float, float]:
    """
    Function to calculate Whittle estimate for Hurst parameter
    Code taken from https://github.com/JFBazille/ICode/blob/master/ICode/estimators/whittle.py
        :param data: 2D array containing samples of Fractional Brownian Noise as rows
        :param idx: Sample index to estimate hurst from
        :return: Estimate of Hurst parameter
    """
    datap = data[idx, :]
    N = datap.shape[0]
    halfN = int((N - 1) / 2)
    tmp = np.abs(np.fft.fft(datap))
    gamma_hat = np.exp(2 * np.log(tmp[1:halfN + 1])) / (2 * np.pi * N)
    func = lambda Hurst: whittle_ll(Hurst, gamma_hat, N)
    return idx, float(so.fminbound(func, 0., 1.))


def estimate_hurst(true: np.ndarray, synthetic: np.ndarray, exp_dict: dict, S: int, config: ConfigDict) -> dict:
    """
    Function to estimate Hurst index from dataset
        :param true: Exact samples
        :param synthetic: Synthetic Samples
        :param exp_dict: Experiment dictionary
        :param S: Number of samples
        :param config: ML experiment configuration file
        :return: Updated experiment dictionary
    """
    gen_data = reduce_to_fBn(synthetic, reduce=True)
    true_data = reduce_to_fBn(true, reduce=True)
    true_Hs = []
    synth_Hs = []
    for j in tqdm(range(S), dynamic_ncols=False, desc="Estimating Hurst Parameter ::", position=0):
        true_Hs.append(optimise_whittle(data=true_data, idx=j))
        synth_Hs.append(optimise_whittle(data=gen_data, idx=j))
    exp_dict[config.exp_keys[11]] = true_Hs
    exp_dict[config.exp_keys[12]] = synth_Hs
    print("Exact samples :: Mean {}, Std {}".format(np.mean(true_Hs), np.std(true_Hs)))
    print("Synthetic samples :: Mean {}, Std {}".format(np.mean(synth_Hs), np.std(synth_Hs)))
    return exp_dict


def compute_pvals(forward_samples: np.ndarray, reverse_samples: np.ndarray) -> list:
    """
    Function to compute KS test p values between marginals
        :param forward_samples: Exact samples
        :param reverse_samples: Final time reverse-diffusion samples
        :return: List of p-values
    """
    assert (forward_samples.shape == reverse_samples.shape)
    ps = []
    timeDim = forward_samples.shape[1]
    for t in np.arange(start=0, stop=timeDim, step=1):
        forward_t = forward_samples[:, t].flatten()
        reverse_samples_t = reverse_samples[:, t].flatten()
        ks_res = kstest(forward_t, reverse_samples_t)
        ps.append(ks_res[1])
        print("KS-test statistic for marginal at time {} :: {}".format(t, ks_res))
    return ps


def time_aug(data_samples: torch.Tensor, time_ax: torch.Tensor) -> torch.Tensor:
    """
    Augment 1-dimensional time series with a monotonically increasing time dimension
        :param data_samples: Original 1-dimensional time series
        :param time_ax: Real time process evolves in
        :return: Time augmented time series
    """
    N, T, d = data_samples.shape
    assert (time_ax.shape == (T, 1))
    timeaug = torch.stack([torch.column_stack([time_ax, data_samples[i, :, :]]) for i in range(N)], dim=0).to(
        time_ax.device)
    assert (timeaug.shape == (N, T, d + 1))
    return timeaug


def invisibility_reset(timeaug: torch.Tensor, ts_dim: int) -> torch.Tensor:
    """
        Transform time augmented time series with invisibility reset
        :param timeaug: Time augmented time series
        :param ts_dim: Original time series dimensions
        :return: Transformed time series
    """
    N, T = timeaug.shape[:2]
    assert (timeaug.shape[-1] == ts_dim + 1)
    # Wi = torch.hstack([timeaug[:, [0], :], torch.zeros_like(timeaug[:, [0], :]), torch.diff(timeaug, dim=1),
    #                torch.zeros_like(timeaug[:, [-1], :]), -timeaug[:, [-1], :]])
    # assert (Wi.shape == (N, T + 3, ts_dim + 1))
    Wi = torch.hstack([timeaug[:, [0], :], torch.diff(timeaug, dim=1)])
    assert (Wi.shape == (N, T, ts_dim + 1))
    # assert (torch.sum(np.abs(torch.sum(torch.sum(Wi, dim=2), dim=1)), dim=0) < 1e-10)
    return Wi


def compute_signature(sample: torch.Tensor, trunc: int, interval: rhpy.Interval, dim: int, coefftype: rhpy.ScalarMeta):
    # To work with RoughPy, we first need to first construct a context
    CTX = rhpy.get_context(width=dim, depth=trunc,
                           coeffs=coefftype)  # (Transformed TS dimension, Signature Truncation, TS DataType)

    # Given a context, we need to transform our data into a stream of increments
    stream = rhpy.LieIncrementStream.from_increments(data=sample, ctx=CTX)

    # Now compute the signature over the whole time span TODO: HOW DO WE DEAL WITH INVISIBILITY AUGMENTATION IN TIME
    #  DIMENSION?
    sig = torch.Tensor(np.array(stream.signature(interval))).to(sample.device)  # TODO: What is resolution?
    if dim > 1:
        assert (sig.shape[0] == ((np.power(dim, trunc + 1) - 1) / (dim - 1)))
    else:
        assert (sig.shape[0] == trunc + 1)
    return sig


def assert_chen_identity(sample: np.ndarray, trunc: int, dim: int, coefftype: rhpy.ScalarMeta) -> None:
    """
    Sanity check to ensure Chen's identity is preserved
    :param sample: Single time series sample
    :param trunc: Signature truncation level
    :param dim: Time series dimensionality
    :param coefftype: Time series data type
    :return: AssertionError or None
    """
    assert (len(sample.shape) == 2)
    T = sample.shape[0]
    intv1 = rhpy.RealInterval(0, T // 2)
    sig1 = compute_signature(sample=sample, trunc=trunc, dim=dim, interval=intv1, coefftype=coefftype)
    intv2 = rhpy.RealInterval(T // 2, T)
    sig2 = compute_signature(sample=sample, trunc=trunc, dim=dim, interval=intv2, coefftype=coefftype)
    intv3 = rhpy.RealInterval(0, T)
    sig3 = compute_signature(sample=sample, trunc=trunc, dim=dim, interval=intv3, coefftype=coefftype)
    assert (np.all(sig1.shape == sig2.shape) and np.all(sig2.shape == sig3.shape))
    assert (np.all(np.abs(sig3 - tensor_algebra_product(sig1=sig1, sig2=sig2, dim=dim, trunc=trunc)) < 1e-6))


def ts_signature_pipeline(data_batch: torch.Tensor, trunc: int, times: torch.Tensor, interval=None) -> torch.Tensor:
    """
    Pipeline to compute the signature at each time for each sample of data batch
        :param data_batch: Data of shape (NumSamples, TSLength, TSDims)
        :param trunc: Signature truncation level
        :return: Signature for each time
    """
    assert (len(data_batch.shape) == 3 and len(times.shape) == 2)
    N, T, d = data_batch.shape
    timeaug = time_aug(data_batch, times[:T, :])
    transformed = invisibility_reset(timeaug, ts_dim=d)
    dims = transformed.shape[-1]
    if interval is None: interval = rhpy.RealInterval(0, T)
    feats = torch.stack([compute_signature(sample=transformed[i, :, :], trunc=trunc, interval=interval,
                                           dim=dims, coefftype=rhpy.DPReal) for i in range(N)], dim=0)
    assert (feats.shape == (N, compute_sig_size(dim=dims, trunc=trunc)))
    return feats


def compute_sig_size(dim: int, trunc: int) -> int:
    """
    Compute the number of elements for a truncated signature
        :param dim: Dimension of augmented time series
        :param trunc: Truncation level of signature
        :return: Number of elements in the truncated signature
    """
    if dim > 1:
        return int(np.power(dim, trunc + 1) - 1 / (dim - 1))
    else:
        return int(trunc + 1)


def tensor_algebra_product(sig1: torch.Tensor, sig2: torch.Tensor, dim: int, trunc: int) -> torch.Tensor:
    """
    Manually compute Chen's identity over two non-overlapping consecutive intervals
        :param sig1: Signature over first interval
        :param sig2: Signature over second interval
        :param dim: Dimension of augmented time series
        :param trunc: Signature truncation level
        :return: Signature of concatenated path
    """
    assert (len(sig1.shape) == len(sig2.shape) == 1 and sig1.shape == sig2.shape and sig1[0] == sig2[
        0] == 1. and 1 <= trunc <= 3)
    device = sig1.device
    product = torch.zeros_like(sig1)
    # For trunc 0: Constant of 1
    product[0] = sig1[0] * sig2[0]
    # For trunc 1: Compute outer product
    product[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)] = (
            sig1[0] * sig2[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)] + sig2[0] * sig1[
                                                                                                              compute_sig_size(
                                                                                                                  dim=dim,
                                                                                                                  trunc=0):compute_sig_size(
                                                                                                                  dim=dim,
                                                                                                                  trunc=1)]).flatten()
    if trunc > 1:
        # For trunc 2: First compute cross terms
        level2 = torch.reshape(torch.atleast_2d(
            sig1[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)]).T @ torch.atleast_2d(
            sig2[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)]), (dim ** 2,))
        # For trunc 2: Compute outer products
        level2 += sig1[0] * sig2[compute_sig_size(dim=dim, trunc=1):compute_sig_size(dim=dim, trunc=2)] + sig2[
            0] * sig1[
                 compute_sig_size(dim=dim, trunc=1):compute_sig_size(dim=dim,
                                                                     trunc=2)]
        product[compute_sig_size(dim=dim, trunc=1):compute_sig_size(dim=dim, trunc=2)] = level2
        if trunc > 2:
            # For trunc 3: First the outer product
            level3 = sig1[0] * sig2[compute_sig_size(dim=dim, trunc=2):compute_sig_size(dim=dim, trunc=3)] + sig2[
                0] * sig1[
                     compute_sig_size(dim=dim, trunc=2):compute_sig_size(dim=dim,
                                                                         trunc=3)]
            # For trunc 3: Next the cross terms of 1_i,2_jk
            level3 += torch.reshape(torch.atleast_2d(
                sig1[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)]).T @ torch.atleast_2d(
                sig2[compute_sig_size(dim=dim, trunc=1):compute_sig_size(dim=dim, trunc=2)]), (dim ** 3,))
            # For trunc 3: Next the cross terms of 1_ij,2k (note we work in a non-commutative basis)
            level3 += torch.reshape(torch.atleast_2d(
                sig1[compute_sig_size(dim=dim, trunc=1):compute_sig_size(dim=dim, trunc=2)]).T @ torch.atleast_2d(
                sig2[compute_sig_size(dim=dim, trunc=0):compute_sig_size(dim=dim, trunc=1)]), (dim ** 3,))
            product[compute_sig_size(dim=dim, trunc=2):compute_sig_size(dim=dim, trunc=3)] = level3
    assert (product.shape == (compute_sig_size(dim=dim, trunc=trunc),))
    return torch.atleast_2d(product)
