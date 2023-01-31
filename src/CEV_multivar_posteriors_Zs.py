from tqdm import tqdm

from utils.math_functions import np, truncnorm, snorm, sinvgamma



def sigmaX_MH_Zs(priorParams, transformedVols, currSigmaX, alpha, muX, deltaT, invfBnCovMat, X0, N, rng):
    alpha0, beta0 = priorParams
    ZN1 = transformedVols[:N]
    diffZ = transformedVols[1:] - ZN1
    b = np.reshape(diffZ + alpha * deltaT, (N, 1))
    currF = np.reshape(alpha * muX * deltaT * np.power(X0, -1) * np.exp(-currSigmaX * ZN1) - 0.5 * currSigmaX * deltaT,
                       (N, 1))
    # Propose new parameter
    proposalScale = 3.
    newSigmaX2 = (snorm.rvs(loc=currSigmaX ** 2, scale=proposalScale))
    # Prior likelihood
    logAccProb = sinvgamma.logpdf(x=newSigmaX2, a=alpha0, scale=beta0) - sinvgamma.logpdf(x=currSigmaX ** 2, a=alpha0,
                                                                                          scale=beta0)
    if logAccProb == -np.inf:
        return currSigmaX, 1
    else:
        newF = np.reshape(alpha * muX * deltaT * np.power(X0, -1) * np.exp(-np.sqrt(newSigmaX2) * ZN1) - 0.5 * np.sqrt(
            newSigmaX2) * deltaT, (N, 1))
        # Likelihood
        logAccProb = -0.5 * (newF.T @ invfBnCovMat @ newF - currF.T @ invfBnCovMat @ currF) + b.T @ invfBnCovMat @ (
                    newF - currF)
        # Proposal Likelihood
        logAccProb += truncnorm.logpdf(x=currSigmaX ** 2, a=-newSigmaX2 / proposalScale, b=np.inf, loc=newSigmaX2,
                                       scale=proposalScale)
        logAccProb -= truncnorm.logpdf(x=newSigmaX2, a=-currSigmaX ** 2 / proposalScale, b=np.inf, loc=currSigmaX ** 2,
                                       scale=proposalScale)
        u = rng.uniform(low=0., high=1.)
        if np.log(u) <= min(0., logAccProb):
            return np.sqrt(newSigmaX2), 1
        return currSigmaX, 0


def obs_mean_posterior(priorParams, obs, vols, deltaT, N, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + np.sum(
        (deltaT * np.diff(obs) * invSigmaIs)) + 0.5 * deltaT * N  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def fBn_covariance(lag, H):
    return 0.5 * (np.power(abs(lag + 1), (2 * H)) + np.power(abs(lag - 1), (2 * H)) - 2 * np.power(abs(lag), (2 * H)))


def fBn_covariance_matrix(N, H):
    """ Covariance matrix for unit increments """
    arr = np.empty(shape=(N, N))  # Pre-allocate memory for speed
    for n in tqdm(range(N)):
        for m in range(n, N):
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


def alpha_MH(priorParams, transformedVols, currAlpha, sigmaX, deltaT, muX, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - 0.5 * sigmaX * deltaT, (N, 1))
    b = np.reshape(muX * deltaT * suff - deltaT, (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b
    d2 = partialD @ (ZN - a)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Propose new parameter
    proposalScale = 3.
    newAlpha = truncnorm.rvs(a=-currAlpha / proposalScale, b=np.inf, loc=currAlpha, scale=proposalScale)
    # Likelihood
    logAccProb = snorm.logpdf(newAlpha, loc=llMean, scale=llStd) - snorm.logpdf(currAlpha, loc=llMean, scale=llStd)
    # Prior likelihood
    logAccProb += truncnorm.logpdf(x=newAlpha, a=-priorMean / priorScale, b=np.inf, loc=priorMean, scale=priorScale)
    logAccProb -= truncnorm.logpdf(x=currAlpha, a=-priorMean / priorScale, b=np.inf, loc=priorMean, scale=priorScale)
    # Proposal likelihood
    logAccProb += truncnorm.logpdf(x=currAlpha, a=-newAlpha / proposalScale, b=np.inf, loc=newAlpha,
                                   scale=proposalScale) - truncnorm.logpdf(
        x=newAlpha, a=-currAlpha / proposalScale, b=np.inf, loc=currAlpha, scale=proposalScale)
    u = rng.uniform(low=0., high=1.)
    if np.log(u) <= min(0., logAccProb):
        return newAlpha, 1
    return currAlpha, 0


def muX_MH(priorParams, transformedVols, currVolMean, alpha, deltaT, sigmaX, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - alpha * deltaT - 0.5 * sigmaX * deltaT, (N, 1))
    b = np.reshape(alpha * deltaT * suff, (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b
    d2 = partialD @ (ZN - a)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Propose new parameter with TruncNorm_{0}(muX, 1.)
    proposalScale = 3.
    newVolMean = truncnorm.rvs(a=-currVolMean / proposalScale, b=np.inf, loc=currVolMean, scale=proposalScale)
    # Likelihood
    logAccProb = snorm.logpdf(newVolMean, loc=llMean, scale=llStd) - snorm.logpdf(currVolMean, loc=llMean, scale=llStd)
    # Prior likelihood
    logAccProb += truncnorm.logpdf(x=newVolMean, a=-priorMean / priorScale, b=np.inf, loc=priorMean, scale=priorScale)
    logAccProb -= truncnorm.logpdf(x=currVolMean, a=-priorMean / priorScale, b=np.inf, loc=priorMean, scale=priorScale)
    # Proposal Likelihood
    logAccProb += truncnorm.logpdf(x=currVolMean, a=-newVolMean / proposalScale, b=np.inf, loc=newVolMean,
                                   scale=proposalScale) - truncnorm.logpdf(
        x=newVolMean, a=-currVolMean / proposalScale, b=np.inf, loc=currVolMean, scale=proposalScale)
    u = rng.uniform(low=0., high=1.)
    if np.log(u) <= min(0., logAccProb):
        return newVolMean, 1
    return currVolMean, 0


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
    return newAlpha, 1


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
    return newVolMean, 1


def sigmaX_MH(priorParams, vols, currSigmaX, alpha, muX, deltaT, V_mat, N, rng):
    alpha0, beta0 = priorParams
    diffX = np.diff(vols)
    dashMuN = alpha * deltaT * np.reshape(muX - vols[:N], (N, 1))
    currDriftless = np.reshape(diffX / currSigmaX, (N, 1)) - dashMuN
    # Propose new parameter
    proposalScale = 2.
    newSigmaX = np.sqrt(
        truncnorm.rvs(a=-currSigmaX ** 2 / proposalScale, b=np.inf, loc=currSigmaX ** 2, scale=proposalScale))
    # Prior likelihood
    logAccProb = sinvgamma.logpdf(x=newSigmaX ** 2, a=alpha0, scale=beta0) - sinvgamma.logpdf(x=currSigmaX ** 2,
                                                                                               a=alpha0, scale=beta0)
    if logAccProb == -np.inf:
        return currSigmaX, 0
    else:
        newDriftless = np.reshape(diffX / newSigmaX, (N, 1)) - dashMuN
        # Likelihood
        logAccProb += -N * (np.log(newSigmaX) - np.log(currSigmaX)) - 0.5 * (
                newDriftless.T @ V_mat @ newDriftless - currDriftless.T @ V_mat @ currDriftless)
        # Proposal Likelihood
        logAccProb += truncnorm.logpdf(x=currSigmaX ** 2, a=-newSigmaX ** 2 / proposalScale, b=np.inf, loc=newSigmaX ** 2,
                                       scale=proposalScale)
        logAccProb -= truncnorm.logpdf(x=newSigmaX ** 2, a=-currSigmaX ** 2 / proposalScale, b=np.inf, loc=currSigmaX ** 2,
                                       scale=proposalScale)
        u = rng.uniform(low=0., high=1.)
        if np.log(u) <= min(0., logAccProb):
            return newSigmaX, 1
        return currSigmaX, 0

def posteriors(muUParams, alphaParams, muXParams, sigmaXParams, deltaT, observations, latents, transformedLatents,
               theta, X0, alphaAcc, volAcc, sigmaXAcc,
               invfBnCovMat=None, V=None,
               rng=np.random.default_rng()):
    muU, alpha, muX, sigmaX, H = theta
    N = latents.shape[0] - 1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, N=N, rng=rng)
    suff = np.power(latents[:N], -1)
    newAlpha, isAcc = alpha_Gibbs(priorParams=alphaParams, transformedVols=transformedLatents,
                                  sigmaX=sigmaX, deltaT=deltaT,
                                  muX=muX, invfBnCovMat=invfBnCovMat, suff=suff, N=N, rng=rng)
    alphaAcc += isAcc
    newVolMean, isAcc = muX_Gibbs(priorParams=muXParams, transformedVols=transformedLatents,
                                  alpha=newAlpha, deltaT=deltaT, sigmaX=sigmaX, suff=suff, invfBnCovMat=invfBnCovMat,
                                  N=N,
                                  rng=rng)
    volAcc += isAcc
    if V is None:
        V = generate_V_matrix(vols=latents, sigmaX=1., N=N, H=H, deltaT=deltaT, invfBnCovMat=invfBnCovMat)
    assert (V.shape[0] == V.shape[1] and V.shape[0] == N)
    newSigmaX, isAcc = sigmaX_MH(priorParams=sigmaXParams, vols=latents, currSigmaX=sigmaX,
                                 alpha=alpha, muX=muX, deltaT=deltaT, V_mat=V,
                                 N=N,
                                 rng=rng)
    sigmaXAcc += isAcc
    return np.array([newObsMean, newAlpha, newVolMean, newSigmaX, H]), alphaAcc, volAcc, sigmaXAcc
