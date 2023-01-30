from tqdm import tqdm

from src.CIR_posteriors import posteriors
from src.ClassFractionalCEV import FractionalCEV
from src.ClassParticleFilter import FractionalParticleFilter
from src.priors import prior
from utils.math_functions import np, beta
from utils.plotting_functions import plot_subplots


def hurst_proposal(H, rng=np.random.default_rng()):
    return rng.beta(a=np.power(H, 2) * np.power(1. - H, -1), b=H)  # alpha = H/(1-H) *beta


def test_no_H(S=10000, muU=1., muX=2., gamma=2., X0=1., U0=0., H=0.8, N=2 ** 7, T=1e-6 * 2 ** 7, nParticles=100,
              rng=np.random.default_rng()):
    sigmaX = np.sqrt(muX * gamma / 0.55)
    deltaT = T / N
    m = FractionalCEV(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    L = Us.size
    plot_subplots(np.arange(0, T + deltaT, step=deltaT), [Xs, Us], [None, None], ["Time", "Time"],
                  ["Volatility", "Log Price"],
                  "Project Model Simulation")
    muUParams, gammaParams, muXParams, sigmaXParams = (0., 0.1), 1., (1.), (2.1, np.power(1., 2) * 1.1)
    thetaPrev = prior(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams, sigmaXParams=sigmaXParams)
    pf = FractionalParticleFilter(nParticles=nParticles, model=m, H=thetaPrev[4], deltaT=deltaT, N=N, rng=rng)
    for j in tqdm(range(1, L)):
        # The index now represents actual time
        pf.run_filter(observation=Us[j], deltaT=deltaT, index=j)
    logLPrev = pf.logLEstimate
    # TODO: Loop contents should go inside ParticleMCMC Class
    for i in range(S):
        # TODO: Sample new theta from proposal (block/Gibbs/independent)
        thetaNew = posteriors(muUParams=muUParams, gammaParams=gammaParams, muXParams=muXParams,
                              sigmaXParams=sigmaXParams, deltaT=deltaT, observations=Us, latents=Xs, theta=thetaPrev,
                              rng=rng)
        thetaNew[-1] = hurst_proposal(H)
        # TODO: Run PF with new theta
        m.__init__(muU=thetaNew[0], gamma=thetaNew[1], muX=thetaNew[2], sigmaX=thetaNew[3], X0=X0, U0=U0)
        pf.__init__(nParticles=nParticles, model=m, H=thetaNew[4], deltaT=deltaT, N=N)
        for j in tqdm(range(1, L)):
            # The index now represents actual time
            # TODO: ENSURE THAT AT LEAST ONE OF THE PARTICLES CONTAINS THE PREVIOUSLY GENERATED PATH
            pf.run_filter(observation=Us[j], deltaT=deltaT, index=j)
        logLNew = pf.logLEstimate
        logAccProb = logLNew - logLPrev
        # TODO: calculate difference between proposal densities between old|new and new|old in log domain
        logAccProb += beta.logpdf(a=np.power(thetaNew[-1], 2) * np.power(1. - thetaNew[-1], -1),
                                  b=thetaNew[-1]) - beta.logpdf(
            a=np.power(thetaPrev[-1], 2) * np.power(1. - thetaPrev[-1], -1), b=thetaPrev[-1])
        u = rng.uniform(0., 1.)
        if u <= min(0., logAccProb):
            thetaPrev = thetaNew
            logLPrev = logLNew

    return thetaPrev, logLPrev

    # TODO: Return final theta and log likelihood of data


test_no_H()
