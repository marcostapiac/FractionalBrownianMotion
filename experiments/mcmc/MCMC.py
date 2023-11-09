import numpy as np
from scipy.stats import gamma as gammaDist
from scipy.stats import norm as snorm

from src.classes.ClassLogSVModel import LogSVModel
from utils.math_functions import acf
from utils.plotting_functions import plot_subplots


def calculate_ACF(data):
    assert (len(data.shape) == 1)
    return acf(data)


def plot_traces(nIters, a, b, s, title):
    iters = np.linspace(0, nIters, nIters)
    xlabels = np.array(["Iterations" for _ in range(3)])
    ylabels = np.array(["a", "b", "s"])
    label_args = np.array([None for _ in range(3)])
    plot_subplots(iters, np.array([a, b, s]), label_args=label_args, xlabels=xlabels, ylabels=ylabels,
                  globalTitle=title)


def preprocessing(logvol2s):
    T = logvol2s.shape[0] - 2
    Z = np.atleast_2d([[1., h] for h in logvol2s[:T]]).T  # Z contains h0, ... hT-1
    H1 = np.atleast_2d(logvol2s[1:T + 1]).T  # Contains h1:hT
    assert (Z.shape[1] == H1.shape[0] and H1.shape[1] == 1)
    return Z, H1


def get_prior_parameters(isStationary=True):
    if isStationary:
        return -0.7, 0.01, 1., 1.
    return np.atleast_2d([-0.723, 0.9]).T, 0.1 * np.eye(N=2, M=2), 4.8, .5


def sample_b_prior():
    """ Only to be used if enforcing stationarity """
    return np.random.uniform(-1., 1.)


def sample_b_posterior(Hs, a, b, s):
    """ Only to be used if enforcing stationarity """
    proposed = b + np.random.uniform(low=-0.1, high=0.1)  # np.random.normal(loc=b, scale=0.01)
    if not (-1 <= proposed <= 1):
        return b
    logAccProb = -((b - proposed) / (2 * np.power(s, 2))) * np.sum(
        [-(b + proposed) * np.power(ht1, 2) + 2 * ht1 * (ht - a) for ht1, ht in zip(Hs[:Hs.shape[0] - 1], Hs[1:])])
    u = np.random.uniform(low=0., high=1.)
    if u <= np.exp(min(0., logAccProb)):
        return proposed
    return b


def sample_a_prior(pMean, pCov):
    """ Only to be used if enforcing stationarity """
    return pMean + np.sqrt(pCov) * np.random.normal()


def sample_a_posterior(pMean, pCov, logVol2s, b, s):
    """ Only to be used if enforcing stationarity """
    T = logVol2s.shape[0] - 2
    tildeB = np.atleast_2d([1, -b])
    assert (tildeB.shape[0] == 1 and tildeB.shape[1] == 2)
    tildeH = np.atleast_2d([logVol2s[1:T + 1], logVol2s[:T]])
    assert (tildeH.shape[0] == 2 and tildeH.shape[1] == T and tildeH[1, 0] == logVol2s[0] and tildeH[0, -1] == logVol2s[
        T])
    postCov = np.power((T / np.power(s, 2) + 1. / pCov), -1)
    postMean = (np.squeeze(tildeB @ tildeH @ np.ones(shape=(T, 1))) / np.power(s, 2) + pMean / pCov) * postCov
    return postMean + np.sqrt(postCov) * np.random.normal()


def sample_a_b_prior(pMean, pCov):
    return pMean + np.linalg.cholesky(pCov) @ np.atleast_2d(np.random.normal(size=pCov.shape[0])).T


def sample_a_b_posterior(pMean, pCov, Zs, lnH1s, s):
    """Blocked update to speed convergence and assumes normal prior so Gibbs posterior is normal"""
    # TODO: Optimize inverse computation (2 inverses right now)
    invPriorCov = np.linalg.inv(pCov)
    postCov = np.linalg.inv(invPriorCov + Zs @ Zs.T / np.power(s, 2))
    postMean = postCov @ (invPriorCov @ pMean + Zs @ lnH1s / np.power(s, 2))
    return postMean + np.linalg.cholesky(postCov) @ np.atleast_2d(np.random.normal(size=postCov.shape[0])).T


def sample_s_prior(shape, scale):
    return np.power(gammaDist.rvs(a=shape, scale=1. / scale, size=1), -0.5)[0]


def sample_s_posterior(priorShape, priorScale, Zs, lnH1s, a, b):
    """Assumes IG prior (since s>0) so Gibbs posterior is normal """
    alphaT = np.atleast_2d([a, b])  # (1x2) array
    postShape = priorShape + lnH1s.shape[0] / 2.
    postScale = priorScale + 0.5 * np.sum(np.power((lnH1s - (alphaT @ Zs).T), 2))
    return np.power(gammaDist.rvs(a=postShape, scale=1. / postScale, size=1), -0.5)[0]


def logvol_proposal(currLogVols, rho=0.243):
    """ Pre-conditioned CN proposal for asymmetric proposals"""
    assert (0. < rho < 1.)
    return np.sqrt(1. - np.power(rho, 2)) * currLogVols + rho * np.random.normal(size=currLogVols.shape[0])


def sample_logvols(prices, prevIterLogVols, a, b, s, rho=0.33):
    """ Metropolis within Gibbs step allowing for asymmetric proposal function """
    T = len(prevIterLogVols) - 2  # logVols array contains extra h0, hT+1 terms
    logVols = np.zeros(shape=(T,))
    proposed = logvol_proposal(prevIterLogVols[1:T + 1], rho=rho)
    assert (logVols.shape[0] == proposed.shape[0])
    nextTimeLogVols = prevIterLogVols[2:]
    prevTimeLogVols = prevIterLogVols[:T]
    assert (proposed.shape[0] == prevIterLogVols[1:T + 1].shape[0] and nextTimeLogVols.shape[0] ==
            prevTimeLogVols.shape[0])
    logPropProb = snorm.logpdf(proposed, loc=np.sqrt(1. - np.power(rho, 2)) * prevIterLogVols[1:T + 1],
                               scale=rho) - snorm.logpdf(prevIterLogVols[1:T + 1],
                                                         loc=np.sqrt(1. - np.power(rho, 2)) * proposed,
                                                         scale=rho)  # Should be 0?
    logLLProb = snorm.logpdf(prices, loc=0., scale=np.exp(0.5 * proposed)) - snorm.logpdf(prices, loc=0.,
                                                                                          scale=np.exp(
                                                                                              0.5 * prevIterLogVols[
                                                                                                    1:T + 1]))
    logLLProb += snorm.logpdf(proposed, loc=a + b * prevTimeLogVols, scale=s) - snorm.logpdf(
        prevIterLogVols[1:T + 1],
        loc=a + b * prevTimeLogVols,
        scale=s)
    logLLProb += snorm.logpdf(nextTimeLogVols, loc=a + b * proposed, scale=s) - snorm.logpdf(nextTimeLogVols,
                                                                                             loc=a + b * prevIterLogVols[
                                                                                                         1:T + 1],
                                                                                             scale=s)
    logAccProb = logLLProb + logPropProb
    """ Accept-reject step"""
    u = np.random.uniform(size=logAccProb.shape[0])
    logAccProb = np.where(logAccProb > 0., 0., logAccProb)
    indxs = np.where(u - np.exp(logAccProb) <= 0., True, False)
    logVols[indxs] = proposed[indxs]
    logVols[np.logical_not(indxs)] = prevIterLogVols[1:T + 1][np.logical_not(indxs)]
    """ We use bwd and fwd forecasting to compute h(T+1) to avoid assigning priors on logVols  """
    lrvar = a / (1. - b)
    logVols = np.insert(logVols, 0, np.power(b, 2) * (prevIterLogVols[2] - lrvar) + lrvar)  # E[h0|F(2,infty)]
    logVols = np.append(logVols, a + b * (a + b * prevIterLogVols[T - 1]))  # E[hT+1|F(-infty, T)]
    assert (logVols.shape[0] == T + 2)
    return logVols


def mcmc(nIters, isStationary=True):
    model = LogSVModel(a=-0.736, b=0.9, s=0.363, isStationary=isStationary)
    df = model.simulate_obs(9999)
    prices = df.loc[:, "price"]
    T = prices.shape[0]
    lv2s = np.zeros(shape=(T + 2,))
    lv2s[1:T + 1] = LogSVModel(a=-0.736, b=0.9, s=0.363, isStationary=isStationary).simulate_log_vols(prices.shape[0])
    priorMean, priorCov, alpha0, beta0 = get_prior_parameters(isStationary=isStationary)
    if not isStationary:
        a, b = sample_a_b_prior(pMean=priorMean, pCov=priorCov)[:, 0]
    else:
        a = sample_a_prior(priorMean, priorCov)
        b = sample_b_prior()
    s = sample_s_prior(shape=alpha0, scale=beta0)
    print(0)
    print(a, b, s)
    las = [a]
    bs = [b]
    ss = [s]
    """ Initialised from init param using logVol model """
    lrvar = a / (1. - b)
    lv2s[0] = np.power(b, 2) * (lv2s[2] - lrvar) + lrvar
    lv2s[-1] = a + b * a + np.power(b, 2) * lv2s[-3]
    """ MCMC starts now --> how to deal with slow AND overflow (overflow due to stationarity issue?) """
    for i in range(1, nIters):
        lv2s = sample_logvols(prices, prevIterLogVols=lv2s, a=las[i - 1], b=bs[i - 1], s=ss[i - 1])
        Z, H1 = preprocessing(lv2s)
        if isStationary:
            a = sample_a_posterior(pMean=priorMean, pCov=priorCov, logVol2s=lv2s, b=bs[i - 1], s=ss[i - 1])
            b = sample_b_posterior(lv2s[:lv2s.shape[0] - 1], a=a, b=b, s=ss[i - 1])
        else:
            a, b = sample_a_b_posterior(pMean=priorMean, pCov=priorCov, Zs=Z, lnH1s=H1, s=ss[i - 1])[:, 0]
        s = sample_s_posterior(priorShape=alpha0, priorScale=beta0, Zs=Z, lnH1s=H1, a=a, b=b)
        print(a, b, s)
        las.append(a)
        bs.append(b)
        ss.append(s)
    bACF = calculate_ACF(np.array(bs))
    aACF = calculate_ACF(np.array(las))
    sACF = calculate_ACF(np.array(ss))
    ESS = max(min(np.squeeze(np.nonzero((bACF < 1e-3))), default=1),
              min(np.squeeze(np.nonzero((aACF < 1e-3))), default=1),
              min(np.squeeze(np.nonzero((sACF < 1e-3))), default=1))
    plot_traces(np.array(bACF).shape[0], aACF, bACF, sACF, title="Metropolis-within-Gibbs ACF")
    plot_traces(int((nIters) / ESS), las[::ESS], bs[::ESS], ss[::ESS], title="Metropolis-within-Gibbs Trace Plots")
    burnIn = 4000
    print(np.mean(las[burnIn::ESS]), np.mean(bs[burnIn::ESS]), np.mean(ss[burnIn::ESS]))


if "name" == "__main__":
    mcmc(nIters=10000, isStationary=False)
