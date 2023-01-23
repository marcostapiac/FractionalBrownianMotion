from utils.math_functions import np, truncnorm
from scipy import stats


def obs_mean_posterior(priorParams, obs, vols, deltaT, rng):
    mu0, sigma0 = priorParams
    invSigmaIs = np.power(deltaT, -1) * np.exp(-vols[1:])  # 1/sigma_{i}^{2}
    a1 = np.power(sigma0, -2) + np.power(deltaT, 2) * np.sum(invSigmaIs)  # TODO: More efficient through matrix mult?
    a2 = mu0 * np.power(sigma0, -2) + np.sum(
        (deltaT * np.diff(obs) + 0.5 * np.power(deltaT, 2) * np.exp(vols[1:])) * invSigmaIs)  # TODO: Efficiency?
    return (a2 / a1) + np.power(a1, -0.5) * rng.normal()


def vol_meanRev_posterior(priorParams, suff3, suff4, sigmaX, deltaT, H, rng):
    lambd = priorParams
    c1 = suff3 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 2.))
    c2 = suff4 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.))
    postMean = (c2 - lambd) * np.power(c1, -1)
    postStd = np.power(c1, -0.5)
    return truncnorm.rvs(a=-postMean / postStd, b=np.inf, loc=postMean, scale=postStd)


def vol_mean_posterior(priorParams, suff1, suff2, suff6, gamma, sigmaX, deltaT, H):
    eta = priorParams
    d1 = suff6 * np.power(gamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    d2 = suff1 * np.power(gamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
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


def metropolis_hastings(priorParams, suff1, suff2, suff6, latents, N, gamma, muX, sigmaX, deltaT, H, rng):
    current = np.array([gamma, muX]).reshape((2, 1))
    lambd, eta = priorParams
    currSuff3 = np.sum(np.power((muX - latents[:N]) / latents[:N], 2))  # (muX - Xi-1)^2/Xi-1^{2}
    currSuff4 = np.sum(
        (np.diff(latents) * (muX - latents[:N])) / np.power(latents[:N], 2))  # (Xi-Xi-1)(muX - Xi-1)/Xi-1^2
    currC1 = currSuff3 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 2.))
    currC2 = currSuff4 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.))
    currentGammaPostMean = (currC2 - lambd) * np.power(currC1, -1)
    currentGammaPostStd = np.power(currC1, -0.5)
    currD1 = suff6 * np.power(gamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    currD2 = suff1 * np.power(gamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    currD2 += suff2 * gamma * np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.)
    currentMuXPostMean = (currD2 - eta) * np.power(currD1, -1)
    currentMuXPostStd = np.power(currD1, -0.5)

    proposal = current + 0.1*rng.normal(size=2)
    proposalGamma, proposalMuX = proposal[0, 0], proposal[1, 0]
    proposalSuff3 = np.sum(np.power((muX - latents[:N]) / latents[:N], 2))  # (muX - Xi-1)^2/Xi-1^{2}
    proposalSuff4 = np.sum(
        (np.diff(latents) * (muX - latents[:N])) / np.power(latents[:N], 2))  # (Xi-Xi-1)(muX - Xi-1)/Xi-1^2
    proposalC1 = proposalSuff3 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 2.))
    proposalC2 = proposalSuff4 * (np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.))
    proposalGammaPostMean = (proposalC2 - lambd) * np.power(proposalC1, -1)
    proposalGammaPostStd = np.power(proposalC1, -0.5)
    proposalD1 = suff6 * np.power(proposalGamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    proposalD2 = suff1 * np.power(proposalGamma / sigmaX, 2) * np.power(deltaT, -2. * H + 2.)
    proposalD2 += suff2 * proposalGamma * np.power(sigmaX, -2) * np.power(deltaT, -2. * H + 1.)
    proposalMuXPostMean = (proposalD2 - eta) * np.power(proposalD1, -1)
    proposalMuXPostStd = np.power(proposalD1, -0.5)

    #logAccProb = -stats.norm.logpdf(proposalGamma, loc=gamma, scale=.1)+stats.norm.logpdf(gamma, loc=proposalGamma, scale=.1)
    #logAccProb += -(stats.norm.logpdf(proposalMuX, loc=muX, scale=.1))+(stats.norm.logpdf(muX, loc=proposalMuX, scale=.1))
    logAccProb = truncnorm.logpdf(proposalGamma, a=-proposalGammaPostMean / proposalGammaPostStd, b=np.inf,
                                   loc=proposalGammaPostMean, scale=proposalGammaPostStd)
    logAccProb += truncnorm.logpdf(proposalMuX, a=-proposalMuXPostMean / proposalMuXPostStd, b=np.inf,
                                   loc=proposalMuXPostMean, scale=proposalMuXPostStd)
    logAccProb -= truncnorm.logpdf(gamma, a=-currentGammaPostMean / currentGammaPostStd, b=np.inf,
                                   loc=currentGammaPostMean, scale=currentGammaPostStd)
    logAccProb -= truncnorm.logpdf(muX, a=-currentMuXPostMean / currentMuXPostStd, b=np.inf,
                                   loc=currentMuXPostMean, scale=currentMuXPostStd)

    u = rng.uniform(0., 1.)
    if np.log(u) <= min(0., logAccProb):
        return proposalGamma, proposalMuX
    return gamma, muX


def posteriors(muUParams, gammaParams, muXParams, sigmaXParams, deltaT, observations, latents, theta,
               rng=np.random.default_rng()):
    muU, gamma, muX, sigmaX, H = theta
    N = latents.shape[0] - 1
    newObsMean = obs_mean_posterior(priorParams=muUParams,
                                    obs=observations, vols=latents, deltaT=deltaT, rng=rng)
    suff1 = np.sum(1. / latents[:N])  # 1/Xi-1
    suff6 = np.sum(np.power(latents[:N], -2))
    suff2 = np.sum(np.diff(latents) / np.power(latents[:N], 2))  # (Xi-Xi-1)/Xi-1^{2}
    suff5 = np.sum(np.power(np.diff(latents) / latents[:N], 2))  # (Xi-Xi-1)^2/Xi-1^2
    #newVolMean = vol_mean_posterior(priorParams=muXParams, suff1=suff1, suff2=suff2, suff6=suff6, gamma=gamma,
    #                                sigmaX=sigmaX, deltaT=deltaT, H=H)
    #suff3 = np.sum(np.power((newVolMean - latents[:N]) / latents[:N], 2))  # (muX - Xi-1)^2/Xi-1^{2}
    #suff4 = np.sum(
    #    (np.diff(latents) * (newVolMean - latents[:N])) / np.power(latents[:N], 2))  # (Xi-Xi-1)(muX - Xi-1)/Xi-1^2
    #newMeanRev = vol_meanRev_posterior(
    #    priorParams=gammaParams, suff3=suff3, suff4=suff4, sigmaX=sigmaX, deltaT=deltaT, H=H, rng=rng)
    newMeanRev, newVolMean = metropolis_hastings(priorParams=(gammaParams, muXParams), suff1=suff1, suff2=suff2,
                                                 suff6=suff6, latents=latents, N=N, gamma=gamma, muX=muX,
                                                 sigmaX=sigmaX, deltaT=deltaT, H=H, rng=rng)
    suff3 = np.sum(np.power((newVolMean - latents[:N]) / latents[:N], 2))  # (muX - Xi-1)^2/Xi-1^{2}
    suff4 = np.sum(
        (np.diff(latents) * (newVolMean - latents[:N])) / np.power(latents[:N], 2))  # (Xi-Xi-1)(muX - Xi-1)/Xi-1^2

    newSigmaX = vol_sigma_posterior(
        priorParams=sigmaXParams, suff3=suff3, suff4=suff4, suff5=suff5, gamma=newMeanRev, deltaT=deltaT, N=N, H=H,
        rng=rng)
    return np.array([newObsMean, newMeanRev, newVolMean, newSigmaX, H])
