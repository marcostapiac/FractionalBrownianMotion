from numpy import abs, broadcast_to, log, exp
import numpy as np
from scipy.stats import chi2
from scipy.stats import gamma as gammaDist
from scipy.stats import norm as snorm
from scipy.stats import truncnorm
from scipy.stats import beta
from scipy.stats import multivariate_normal as multsnorm
from scipy.optimize import minimize
from scipy.special import gamma as gammafnc
from scipy.stats import invgamma as sinvgamma
from scipy.stats import ncx2
import pandas as pd


def logsumexp(w, h, x, axis=0, isLog=False):
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
    acorr = np.fft.ifft(pwr).real *np.power(np.var(data)* N,-1)
    return acorr
