from utils.math_functions import np, truncnorm


def obs_mean_posterior(priorParams, obs, vols, deltaT, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + deltaT * np.sum(
        (np.diff(obs) + 0.5 * np.exp(vols[1:]) * deltaT) * invSigmaIs)  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def vol_meanRev_posterior(priorParams, suff3, suff4, sigmaX, deltaT, H, rng):
    lambd = priorParams
    c1 = suff3 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 2.))
    c2 = suff4 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.))
    postMean = (c2 - lambd) / c1
    postStd = np.power(c1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def vol_mean_posterior(priorParams, suff1, suff2, gamma, sigmaX, deltaT, H, N, rng):
    eta = priorParams
    d1 = suff1 * np.power(gamma, 2) * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 2.))
    d2 = N * np.power(gamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    d2 += suff2 * gamma * np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.)
    postMean = (d2 - eta) * np.power(d1, -1)
    postStd = np.power(d1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def vol_sigma_posterior(priorParams, suff3, suff4, suff5, N, gamma, deltaT, H, rng):
    alpha, beta = priorParams
    b1 = alpha + N / 2.  # T/2
    b2 = beta + 0.5 * np.power(deltaT, -2 * H) * (
            suff5 - 2 * gamma * deltaT * suff4 + np.power(gamma * deltaT, 2) * suff3)
    return np.sqrt(1. / rng.gamma(shape=b1, scale=1. / b2))


def posteriors(muUParams, gammaParams, muXParams, sigmaXParams, deltaT, observations, latents, theta,
               rng=np.random.default_rng()):
    muU, gamma, muX, sigmaX, H = theta
    N = latents.shape[0] - 1
    suff1 = np.sum(1. / latents[:N])  # 1/Xi-1
    suff2 = np.sum(np.diff(latents) / latents[:N])  # (Xi-Xi-1)/Xi-1
    suff3 = np.sum(np.power(muX - latents[:N], 2) / latents[:N])  # (muX - Xi-1)^2/Xi-1
    suff4 = np.sum((np.diff(latents) * (muX - latents[:N])) / latents[:N])  # (Xi-Xi-1)(muX - Xi-1)/Xi-1
    suff5 = np.sum(np.power(np.diff(latents), 2) / latents[:N])  # (Xi-Xi-1)^2/Xi-1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, rng=rng)
    newSigmaX = vol_sigma_posterior(
        priorParams=sigmaXParams, suff3=suff3, suff4=suff4, suff5=suff5, gamma=gamma, deltaT=deltaT, N=N, H=H,
        rng=rng)
    newMeanRev = vol_meanRev_posterior(
        gammaParams, suff3, suff4, newSigmaX, deltaT, H, rng)
    newVolMean = vol_mean_posterior(muXParams, suff1, suff2, newMeanRev, newSigmaX, deltaT, H, N, rng)
    return np.array([newObsMean, newMeanRev, newVolMean, newSigmaX])
