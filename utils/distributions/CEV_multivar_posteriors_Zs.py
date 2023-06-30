import numpy as np
from numba import njit, prange
from scipy.stats import invgamma as sinvgamma
from scipy.stats import multivariate_normal as smultnorm
from scipy.stats import norm as snorm
from scipy.stats import truncnorm


def obs_mean_posterior(priorParams, obs, vols, deltaT, N, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + np.sum(
        (deltaT * np.diff(obs) * invSigmaIs)) + 0.5 * deltaT * N  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def fBn_covariance(lag, H):
    return 0.5 * (np.power(abs(lag + 1), (2 * H)) + np.power(abs(lag - 1), (2 * H)) - 2 * np.power(abs(lag), (2 * H)))


@njit(parallel=True)
def fBn_covariance_matrix(N, H):
    """ Covariance matrix for unit increments """
    arr = np.empty(shape=(N, N))  # Pre-allocate memory for speed
    for n in (prange(N)):
        for m in prange(n, N):
            arr[m, n] = 0.5 * (
                    np.power(abs(m - n + 1), (2 * H)) + np.power(abs(m - n - 1), (2 * H)) - 2 * np.power(abs(m - n),
                                                                                                         (2 * H)))
            arr[n, m] = arr[m, n]
    return arr


def generate_V_matrix(vols, sigmaX, N, deltaT=None, H=None, invfBnCovMat=None):
    # TODO: SPEED UP THIS GENERATION
    assert (vols.shape[0] == N + 1)
    invSigmaDash = np.power(sigmaX, -1) * np.diag(np.power(vols[:N], -1))
    if invfBnCovMat is None:
        assert (deltaT and H)
        invfBnCovMat = np.linalg.inv(np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H))
    mat = invSigmaDash.T @ invfBnCovMat @ invSigmaDash
    return mat


def alpha_Gibbs(priorParams, transformedVols, sigmaX, deltaT, muX, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - 0.5 * sigmaX * deltaT, (N, 1))
    b = np.reshape(muX * deltaT * suff - deltaT, (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b + np.power(priorScale, -2)
    d2 = partialD @ (ZN - a) + priorMean * np.power(priorScale, -2)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Sample new parameter
    newAlpha = truncnorm.rvs(a=-llMean / llStd, b=np.inf, loc=llMean, scale=llStd)
    return newAlpha


def muX_Gibbs(priorParams, transformedVols, alpha, deltaT, sigmaX, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - alpha * deltaT - 0.5 * sigmaX * deltaT, (N, 1))
    b = np.reshape(alpha * deltaT * suff, (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b + np.power(priorScale, -2)
    d2 = partialD @ (ZN - a) + priorMean * np.power(priorScale, -2)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Propose new parameter with TruncNorm_{0}(muX, 1.)
    newVolMean = truncnorm.rvs(a=-llMean / llStd, b=np.inf, loc=llMean, scale=llStd)
    return newVolMean


def sigmaX_MH(priorParams, vols, currSigmaX, alpha, muX, deltaT, V_mat, N, rng):
    alpha0, beta0 = priorParams
    diffX = np.diff(vols)
    dashMuN = alpha * deltaT * np.reshape(muX - vols[:N], (N, 1))
    currDriftless = np.reshape(diffX / currSigmaX, (N, 1)) - dashMuN
    # Propose new parameter
    proposalScale = 2.
    newSigmaX = snorm.rvs(loc=currSigmaX, scale=proposalScale)
    # newSigmaX = slognorm.rvs(scale=currSigmaX, s=proposalScale)
    if newSigmaX <= 0.:
        return currSigmaX, 0
    else:
        # Prior likelihood
        logAccProb = sinvgamma.logpdf(x=newSigmaX ** 2, a=alpha0, scale=beta0) - sinvgamma.logpdf(x=currSigmaX ** 2,
                                                                                                  a=alpha0, scale=beta0)
        newDriftless = np.reshape(diffX / (newSigmaX), (N, 1)) - dashMuN
        # Likelihood
        logAccProb += -N * (np.log(newSigmaX) - np.log(currSigmaX)) - 0.5 * np.squeeze(
            newDriftless.T @ V_mat @ newDriftless - currDriftless.T @ V_mat @ currDriftless)
        # Proposal Likelihood
        # logAccProb += snorm.logpdf(x=currSigmaX, loc=newSigmaX, scale=proposalScale)
        # logAccProb -= snorm.logpdf(x=newSigmaX, loc=currSigmaX, scale=proposalScale)
        # logAccProb += slognorm.logpdf(x=currSigmaX, scale=newSigmaX, s=proposalScale)
        # logAccProb -= slognorm.logpdf(x=newSigmaX, scale=currSigmaX, s=proposalScale)
        u = rng.uniform(low=0., high=1.)
        if np.log(u) <= min(0., logAccProb):
            return (newSigmaX), 1
        return currSigmaX, 0


def sigmaX_MH_Zs(priorParams, transformedVols, currSigmaX, suff, X0, alpha, muX, deltaT, invfBnCovMat, N, rng):
    alpha0, beta0 = priorParams
    diffZ = np.diff(transformedVols)
    dashMuN = np.reshape(diffZ + alpha * deltaT, (N, 1))
    currF = np.reshape(
        alpha * muX * deltaT * np.power(X0, -1) * np.exp(-currSigmaX * transformedVols[:N]) - 0.5 * currSigmaX * deltaT,
        (N, 1))  # Propose new parameter
    proposalScale = .4
    newSigmaX = snorm.rvs(loc=currSigmaX, scale=proposalScale)
    if newSigmaX <= 0.:
        return currSigmaX, 0
    else:
        # Prior likelihood
        logAccProb = sinvgamma.logpdf(x=newSigmaX ** 2, a=alpha0, scale=beta0) - sinvgamma.logpdf(x=currSigmaX ** 2,
                                                                                                  a=alpha0, scale=beta0)
        newF = np.reshape(alpha * muX * deltaT * np.power(X0, -1) * np.exp(
            -newSigmaX * transformedVols[:N]) - 0.5 * newSigmaX * deltaT, (N, 1))
        # Likelihood
        logAccProb += -0.5 * (
                newF.T @ invfBnCovMat @ newF - currF.T @ invfBnCovMat @ currF) + dashMuN.T @ invfBnCovMat @ (
                              newF - currF)
        u = rng.uniform(low=0., high=1.)
        if np.log(u) <= min(0., logAccProb):
            return newSigmaX, 1
        return currSigmaX, 0


def latents_MH(i, generator, theta, X0, U0, latents, transformedLatents, observations, invfBnCovMat, N, deltaT, rng,
               rho=0.2):
    muU, alpha, muX, sigmaX, H = theta
    """ Get current path """
    currXs = latents
    currZs = transformedLatents
    currGaussIncs = generator.gaussIncs
    """ Update model """
    generator.__init__(muU=muU, alpha=alpha, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    """ PcN proposal for increments """
    # TODO: Try smaller prop var (rho)
    newGaussIncs = np.sqrt(1. - np.power(rho, 2)) * currGaussIncs + rho * rng.normal(size=len(currGaussIncs))  # pCn
    newXs = generator.state_simulation(H=H, N=N, deltaT=deltaT, gaussRvs=newGaussIncs)
    newZs = generator.lamperti(newXs)
    """ Calculate acceptance probability: Observation likelihood """
    newXN1 = np.reshape(newXs[1:], (N,))
    currXN1 = np.reshape(currXs[1:], (N,))
    # logAccProb = smultnorm.logpdf(x=np.exp(-0.5*newXN1)*observations[1:],mean=np.exp(-0.5*newXN1)*(observations[:N]+(muU-0.5*np.exp(newXs[1:]))*deltaT), cov=deltaT*np.eye(N))
    # logAccProb -= smultnorm.logpdf(x=np.exp(-0.5*currXN1)*observations[1:],mean=np.exp(-0.5*currXN1)*(observations[:N]+(muU-0.5*np.exp(currXs[1:]))*deltaT), cov=deltaT*np.eye(N))
    logAccProb = smultnorm.logpdf(x=observations, mean=newXs, cov=deltaT * np.eye(N + 1))
    logAccProb -= smultnorm.logpdf(x=observations, mean=currXs, cov=deltaT * np.eye(N + 1))
    u = rng.uniform(low=0., high=1.)
    if np.log(u) <= min(0., logAccProb):
        return generator, newXs, newZs
    # if i%10 == 0:
    #    plot(np.arange(0., N*deltaT + deltaT, step=deltaT), [newXs, currXs], ["new", "old"], "Time", "Pos", "Title")
    #    plt.show()
    return generator, currXs, currZs


def posteriors(muUParams, alphaParams, muXParams, sigmaXParams, deltaT, observations, latents, transformedLatents,
               theta, X0, sigmaXAcc,
               invfBnCovMat=None, V=None,
               rng=np.random.default_rng()):
    muU, alpha, muX, sigmaX, H = theta
    N = latents.shape[0] - 1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, N=N, rng=rng)
    suff = np.power(latents[:N], -1)
    newAlpha = alpha_Gibbs(priorParams=alphaParams, transformedVols=transformedLatents,
                           sigmaX=sigmaX, deltaT=deltaT,
                           muX=muX, invfBnCovMat=invfBnCovMat, suff=suff, N=N, rng=rng)
    newVolMean = muX_Gibbs(priorParams=muXParams, transformedVols=transformedLatents,
                           alpha=newAlpha, deltaT=deltaT, sigmaX=sigmaX, suff=suff, invfBnCovMat=invfBnCovMat,
                           N=N,
                           rng=rng)
    # if V is None:
    #    V = generate_V_matrix(vols=latents, sigmaX=1., N=N, H=H, deltaT=deltaT, invfBnCovMat=invfBnCovMat)
    # assert (V.shape[0] == V.shape[1] and V.shape[0] == N)
    # TODO: Try multiple iterations of sigmaX (can thin or can store all)

    # newSigmaX, isAcc = sigmaX_MH(priorParams=sigmaXParams, vols=latents, currSigmaX=sigmaX,
    #                             alpha=newAlpha, muX=newVolMean, deltaT=deltaT, V_mat=V,
    #                             N=N,
    #                             rng=rng)

    newSigmaX, isAcc = sigmaX_MH_Zs(priorParams=sigmaXParams, transformedVols=transformedLatents, currSigmaX=sigmaX,
                                    alpha=newAlpha, suff=suff, muX=newVolMean, deltaT=deltaT, invfBnCovMat=invfBnCovMat,
                                    X0=X0,
                                    N=N,
                                    rng=rng)

    sigmaXAcc += isAcc
    return np.array([newObsMean, newAlpha, newVolMean, newSigmaX, H]), sigmaXAcc
