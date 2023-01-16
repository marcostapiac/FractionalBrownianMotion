from utils.math_functions import np, logsumexp, snorm, gammaDist
from src.ClassFractionalCIR import ProjectModel
from src.ClassParticleFilter import ProjectParticleFilter
from src.ClassMCMC import MCMC
from tqdm import tqdm
from p_tqdm import t_map
from functools import partial
from copy import deepcopy
from utils.plotting_functions import plot_subplots, plot, plt
from src.priors import prior
from src.posteriors import posteriors


def hurst_proposal(H, rng=np.random.default_rng()):
    return rng.beta(a=np.power(H, 2) * np.power(1. - H, -1), b=H)  # alpha = H/(1-H) *beta


def test_no_H(S=10000, muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 7, T=2.5, nParticles=100,
              rng=np.random.default_rng()):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    m = ProjectModel(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    L = Us.size
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Project Model Simulation")
    muUParams, gammaParams, muXParams, sigmaXParams = (muU, 0.1), (gamma), (muX), (2.1, sigmaX / 1.1)
    thetaPrev = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    pf = ProjectParticleFilter(nParticles=nParticles, muU=thetaPrev[0], gamma=thetaPrev[1], muX=thetaPrev[2],
                               sigmaX=thetaPrev[3], H=thetaPrev[4], X0=X0, U0=U0, deltaT=deltaT, N=N, rng=rng)
    for j in tqdm(range(1, L)):
        # The index now represents actual time
        pf.run_filter(observation=Us[j], deltaT=deltaT, index=j)
    logLPrev = pf.logLEstimate
    # TODO: Loop contents should go inside ParticleMCMC Class
    for i in range(S):
        # TODO: Sample new theta from proposal (block/Gibbs/independent)
        thetaNew = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                              sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=vols, theta=thetaPrev,
                              rng=rng)
        thetaNew.append(hurst_proposal(H))
        # TODO: Run PF with new theta
        pf.__init__(nParticles=nParticles, muU=thetaNew[0], gamma=thetaNew[1], muX=thetaNew[2],
                    sigmaX=thetaNew[3], H=thetaNew[4], X0=X0, U0=U0, deltaT=deltaT, N=N)
        for j in tqdm(range(1, L)):
            # The index now represents actual time
            pf.run_filter(observation=Us[j], deltaT=deltaT, index=j)
        # TODO: Store new theta likelihood estimate
        logLNew = pf.logLEstimate
        logAccProb = logLNew - logLPrev
        # TODO: calculate difference between proposal densities between old|new and new|old in log domain
        logAccProb += rng.beta.pdf(a=np.power(HNew, 2) * np.power(1. - HNew, -1), b=HNew) - rng.beta.pdf(
            a=np.power(HPrev, 2) * np.power(1. - HPrev, -1), b=HPrev)
        u = rng.uniform(0., 1.)
        if u <= min(0., logAccProb):
            thetaPrev = thetaNew
            logLPrev = logLNew

    return thetaPrev, logLPrev

    # TODO: Return final theta and log likelihood of data


test_no_H()
