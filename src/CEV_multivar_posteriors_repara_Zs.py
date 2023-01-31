from tqdm import tqdm

from utils.math_functions import np, truncnorm, snorm


def obs_mean_posterior(priorParams, obs, vols, deltaT, N, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + np.sum(
        (deltaT * np.diff(obs) * invSigmaIs)) + 0.5 * deltaT * N  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def fBn_covariance(lag, H):
    return 0.5 * (np.power(abs(lag + 1), (2 * H)) + np.power(abs(lag - 1), (2 * H)) - 2 * np.power(abs(lag), (2 * H)))


def sigmaN_matrix(sigmaX, latents):
    return np.diag(sigmaX * latents)


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
    invSigmaDash = np.diag(np.power(sigmaX * vols[:N], -1))
    if invfBnCovMat is None:
        assert (deltaT and H)
        invfBnCovMat = np.linalg.inv(np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H))
    mat = invSigmaDash.T @ invfBnCovMat @ invSigmaDash
    return mat


def gamma_Gibbs(priorParams, transformedVols, sigmaX, deltaT, alpha, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - 0.5 * sigmaX * deltaT + alpha * deltaT * np.power(sigmaX, -1) * suff, (N, 1))
    b = -deltaT * np.power(sigmaX, -1) * np.reshape(np.ones(N), (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b + np.power(priorScale, -2)
    d2 = partialD @ (ZN - a) + priorMean * np.power(priorScale, -2)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Sample new parameter
    newGamma = truncnorm.rvs(a=-llMean / llStd, b=np.inf, loc=llMean, scale=llStd)
    return newGamma, 1


def alpha_Gibbs(priorParams, transformedVols, gamma, deltaT, sigmaX, suff, invfBnCovMat, N, rng):
    priorMean, priorScale = priorParams
    ZN1 = transformedVols[:N]
    ZN = np.reshape(transformedVols[1:], (N, 1))
    a = np.reshape(ZN1 - gamma*np.power(sigmaX,-1) * deltaT - 0.5 * sigmaX * deltaT, (N, 1))
    b = np.reshape(deltaT*np.power(sigmaX,-1) * suff, (N, 1))
    partialD = b.T @ invfBnCovMat
    d1 = partialD @ b + np.power(priorScale, -2)
    d2 = partialD @ (ZN - a) + priorMean * np.power(priorScale, -2)
    llMean = d2 * np.power(d1, -1)
    llStd = np.power(d1, -0.5)
    # Propose new parameter with TruncNorm_{0}(muX, 1.)
    newVolMean = truncnorm.rvs(a=-llMean / llStd, b=np.inf, loc=llMean, scale=llStd)
    return newVolMean, 1


def vol_sigma_posterior(priorParams, vols, alpha, gamma, deltaT, V_matrix, N, rng):
    alpha0, beta0 = priorParams
    b1 = alpha0 + 0.5 * N
    muN = alpha * deltaT + vols[:N] * (1. - gamma * deltaT)
    driftless = np.reshape(vols[1:] - muN, (N, 1))
    b2 = beta0 + 0.5 * np.squeeze(driftless.T @ V_matrix @ driftless)
    return np.sqrt(1. / rng.gamma(shape=b1, scale=1. / b2)), 1


def posteriors(muUParams, gammaParams, transformedLatents, alphaParams, sigmaXParams, deltaT, observations, latents,
               theta, V=None,
               invfBnCovMat=None,
               rng=np.random.default_rng()):
    muU, gamma, alpha, sigmaX, H = theta
    N = latents.shape[0] - 1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, N=N, rng=rng)
    suff = np.power(latents[:N], -1)
    newAlpha, isAcc = alpha_Gibbs(priorParams=alphaParams, transformedVols=transformedLatents, gamma=gamma,
                                  sigmaX=sigmaX, deltaT=deltaT, invfBnCovMat=invfBnCovMat, suff=suff, N=N, rng=rng)
    newVolMeanRev, isAcc = gamma_Gibbs(priorParams=gammaParams, transformedVols=transformedLatents,
                                       alpha=newAlpha, deltaT=deltaT, sigmaX=sigmaX, suff=suff,
                                       invfBnCovMat=invfBnCovMat,
                                       N=N,
                                       rng=rng)
    if V is None:
        V = generate_V_matrix(vols=latents, sigmaX=1., N=N, H=H, deltaT=deltaT, invfBnCovMat=invfBnCovMat)
    assert (V.shape[0] == V.shape[1] and V.shape[0] == N)
    newSigmaX, isAcc = vol_sigma_posterior(priorParams=sigmaXParams, vols=latents,
                                           alpha=newAlpha, gamma=newVolMeanRev, deltaT=deltaT, V_matrix=V,
                                           N=N,
                                           rng=rng)
    return np.array([newObsMean, newVolMeanRev, newAlpha, newSigmaX, H])
