from utils.math_functions import np, truncnorm


def fBn_covariance(lag, H):
    return 0.5 * (abs(lag + 1) ** (2 * H) + abs(lag - 1) ** (2 * H) - 2 * abs(lag) ** (2 * H))


def sigmaN_matrix(sigmaX, vols):
    return np.diag(sigmaX * vols)


def fBn_covariance_matrix(deltaT, N, H):
    return np.atleast_2d([[fBn_covariance((m - n) * deltaT, H) for m in range(N)] for n in range(N)])


def generate_V_matrix(vols, sigmaX, deltaT, N, H):
    # TODO: Check vols is size N + 1
    assert (vols.shape[0] == N + 1)
    SigmaDash = sigmaN_matrix(sigmaX, vols[:N])
    S = fBn_covariance_matrix(deltaT, N, H)
    return np.linalg.inv(SigmaDash @ S @ SigmaDash.T)


def vol_sigma_posterior(priorParams, vols, muX, gamma, deltaT, V_matrix, N, rng):
    # TODO: Check vols is N:0
    alpha, beta = priorParams
    b1 = alpha + np.power(N, 2) / 2.  # T/2
    muN = gamma * muX * deltaT + vols[:N] * (1. - gamma * deltaT)
    driftless = np.reshape(vols[1:] - muN, (N, 1))  # TODO: CHECK IF THE VOLS IS CORRECT
    b2 = beta + 0.5 * np.squeeze(driftless.T @ V_matrix @ driftless)
    return np.sqrt(1. / rng.gamma(shape=b1, scale=1. / b2))


def obs_mean_posterior(priorParams, obs, vols, deltaT, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + np.sum(
        (deltaT * np.diff(obs) + 0.5 * np.power(deltaT, 2) * np.exp(vols[1:])) * invSigmaIs)  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def vol_meanRev_posterior(priorParams, muX, vols, deltaT, V_matrix, N):
    lambd = priorParams
    driftless = muX * deltaT - deltaT * np.reshape(vols[:N], (N, 1))
    assert (driftless.shape[0] == N and driftless.shape[1] == 1)
    partial_c = driftless.T @ V_matrix
    c1 = np.squeeze(partial_c @ driftless)
    c2 = np.squeeze(partial_c @ (np.diff(vols).reshape((N, 1))))
    postMean = (c2 - lambd) * np.power(c1, -1)
    postStd = np.power(c1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def vol_mean_posterior(priorParams, V_matrix, gamma, vols, deltaT, N):
    eta = priorParams
    d1 = np.power(gamma * deltaT, 2) * np.sum(V_matrix)
    d2 = gamma * deltaT * np.sum(V_matrix @ (np.diff(vols) + gamma * deltaT * vols[:N]).reshape((N, 1)),
                                 axis=0)
    postMean = (d2 - eta) * np.power(d1, -1)
    postStd = np.power(d1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def posteriors(muUParams, gammaParams, muXParams, sigmaXParams, deltaT, observations, latents, theta, V=None,
               rng=np.random.default_rng()):
    muU, gamma, muX, sigmaX, H = theta
    N = latents.shape[0] - 1
    if V is None:
        V = generate_V_matrix(vols=latents, sigmaX=sigmaX, deltaT=deltaT, H=H, N=N)
    assert (V.shape[0] == V.shape[1] and V.shape[0] == N)
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, rng=rng)
    newVolMean = vol_mean_posterior(priorParams=muXParams, V_matrix=V, gamma=gamma, vols=latents, deltaT=deltaT, N=N)
    newMeanRev = vol_meanRev_posterior(priorParams=gammaParams, muX=newVolMean, vols=latents, deltaT=deltaT,
                                       V_matrix=V, N=N)
    newSigmaX = vol_sigma_posterior(
        priorParams=sigmaXParams, vols=latents, gamma=newMeanRev, muX=newVolMean, deltaT=deltaT,
        V_matrix=V * np.power(sigmaX, 2),
        rng=rng, N=N)  # TODO: Check Vmatrix
    return np.array([newObsMean, newMeanRev, newVolMean, newSigmaX, H])
