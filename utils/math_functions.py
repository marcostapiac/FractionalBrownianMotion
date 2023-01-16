from numpy import abs, broadcast_to, log, exp
import numpy as np
from scipy.stats import chi2
from scipy.stats import gamma as gammaDist
from scipy.stats import norm as snorm
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal as multsnorm
from scipy.optimize import minimize
from scipy.special import gamma as gammafnc
from dictances import bhattacharyya
import pandas as pd
from statsmodels.tsa.stattools import acf


def logsumexp(w, h, x, axis=0, isLog=False):
    c = np.max(w)
    broad_l = broadcast_to((w - c).flatten(), x.T.shape).T
    if isLog:
        return c + log((exp(broad_l) * h(x)).sum(axis=axis))
    return exp(c) * (exp(broad_l) * h(x)).sum(axis=axis)
