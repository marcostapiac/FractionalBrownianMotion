from utils.math_functions import np


def prior_H(rng=np.random.default_rng()):
    return rng.uniform(0., 1.)


def prior_gamma(params, rng=np.random.default_rng()):
    rate = params
    return rng.exponential(scale=1./rate)


def prior_muX(params, rng=np.random.default_rng()):
    rate = params
    return rng.exponential(scale=1./rate)


def prior_sigmaX(params, rng=np.random.default_rng()):
    alpha, rate = params
    return np.sqrt(1. / rng.gamma(shape=alpha, scale= 1./rate))


def prior_muU(params, rng=np.random.default_rng()):
    mean, sigma = params
    return rng.normal(loc=mean, scale=sigma)


def prior(muUParams, gammaParams, muXParams, sigmaXParams, rng=np.random.default_rng()):
    return np.array([prior_muU(params=muUParams, rng=rng), prior_gamma(params=gammaParams, rng=rng), prior_muX(params=muXParams,
                                                                                                     rng=rng), prior_sigmaX(
        params=sigmaXParams, rng=rng), prior_H(rng=rng)])

