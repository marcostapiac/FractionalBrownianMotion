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
    invSigmaDash = np.diag(np.power(sigmaX * vols[:N],-1))
    if invfBnCovMat is None:
        assert (deltaT and H)
        invfBnCovMat = np.linalg.inv(np.power(deltaT, 2 * H) * fBn_covariance_matrix(N=N, H=H))
    mat = invSigmaDash.T @ invfBnCovMat @ invSigmaDash
    return mat


def vol_sigma_posterior(priorParams, vols, alpha, gamma, deltaT, V_matrix, N, rng):
    alpha0, beta0 = priorParams
    b1 = alpha0 + 0.5 * N
    muN = alpha * deltaT + vols[:N] * (1. - gamma * deltaT)
    driftless = np.reshape(vols[1:] - muN, (N, 1))
    b2 = beta0 + 0.5 * np.squeeze(driftless.T @ V_matrix @ driftless)
    return np.sqrt(1. / rng.gamma(shape=b1, scale=1. / b2))


def vol_alpha_posterior(priorParams, V_matrix, gamma, vols, deltaT, N):
    mu0, sigma0 = priorParams
    d1 = np.power(deltaT, 2) * np.squeeze(np.sum(V_matrix)) + np.power(sigma0,-2)  # TODO: ISSUE WITH VERY SMALL DELTAT LEADS TO LARGE VARIANCE
    driftless = np.reshape(vols[1:] -vols[:N] + gamma * deltaT * vols[:N], (N, 1))
    d2 = mu0 * np.power(sigma0, -2) + deltaT * np.sum(driftless.T@V_matrix, axis=1)
    postMean = (d2) * np.power(d1, -1)
    postStd = np.power(d1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def vol_meanRev_posterior(priorParams, alpha, vols, deltaT, V_matrix, N):
    mu0, sigma0 = priorParams
    XN1 = np.reshape(vols[:N], (N,1))
    driftless = np.reshape(np.diff(vols) - alpha*deltaT, (N,1))
    assert (driftless.shape[0] == N and driftless.shape[1] == 1)
    c1 = np.power(deltaT,2)*np.squeeze(XN1.T@V_matrix@XN1) + np.power(sigma0, -2)
    c2 = mu0 * np.power(sigma0, -2) - deltaT*np.squeeze(driftless.T @ V_matrix @ XN1)
    postMean = c2 * np.power(c1, -1)
    postStd = np.power(c1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)

def posteriors(muUParams, gammaParams, muXParams, sigmaXParams, deltaT, observations, latents, theta, V=None,
               invfBnCovMat=None,
               rng=np.random.default_rng()):
    muU, gamma, alpha, sigmaX, H = theta
    N = latents.shape[0] - 1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, N=N, rng=rng)
    if V is None:
        V = generate_V_matrix(vols=latents, sigmaX=1., N=N, H=H, deltaT=deltaT, invfBnCovMat=invfBnCovMat)
    assert (V.shape[0] == V.shape[1] and V.shape[0] == N)
    newSigmaX = vol_sigma_posterior(
        priorParams=sigmaXParams, vols=latents, gamma=gamma, alpha=alpha, deltaT=deltaT,
        V_matrix=V,
        rng=rng, N=N)
    V *= np.power(newSigmaX, -2)
    newAlpha = vol_alpha_posterior(priorParams=muXParams, V_matrix=V, gamma=gamma, vols=latents, deltaT=deltaT, N=N)
    newMeanRev = vol_meanRev_posterior(priorParams=gammaParams, alpha=newAlpha, vols=latents, deltaT=deltaT,
                                       V_matrix=V, N=N)
    return np.array([newObsMean, newMeanRev, newAlpha, newSigmaX, H])
