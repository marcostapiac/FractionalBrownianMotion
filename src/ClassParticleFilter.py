from utils.math_functions import np, logsumexp, snorm
from src.ClassProjectModel import ProjectModel
from src.ClassMCMC import pCnMCMC
from tqdm import tqdm
from p_tqdm import t_map
from functools import partial
from copy import deepcopy


class Particle:
    def __init__(self):
        pass

    def state_increment(self, **kwargs):
        pass

    def weight_increment(self, *kwargs):
        pass


class ParticleFilter:
    def __init__(self, nParticles, resampleLimit, rng=np.random.default_rng()):
        self.M = nParticles
        self.rLim = resampleLimit
        self.logWghts = np.array([-np.log(nParticles) for _ in range(nParticles)])
        self.particles = [Particle() for _ in range(nParticles)]
        self.rng = rng

    def compute_likelihood(self):
        return logsumexp(self.logWghts, lambda x: 1., np.ones(self.M), isLog=False)

    def normalise_weights(self):
        lsum_weights = logsumexp(self.logWghts, lambda x: 1., np.ones(self.M), isLog=True)
        return self.logWghts - lsum_weights

    def resample(self):
        """ Adapted from filterpy.monte_carlo.stratified_resample """
        nP = self.M
        u = np.zeros((nP, 1))
        c = np.cumsum(np.exp(self.logWghts))
        c[-1] = 1.0
        i = 0
        u[0] = self.rng.random() / nP
        new_ps = [list([]) for _ in range(nP)]
        for j in range(nP):

            u[j] = u[0] + j / nP

            while u[j] > c[i]:
                i = i + 1

            new_ps[j] = deepcopy(self.particles[i])

        logWeights = np.array([-np.log(nP)] * nP)
        return np.atleast_2d(logWeights), new_ps


class ProjectParticleFilter(ParticleFilter):
    def __init__(self, nParticles, resampleLimit, muU, muX, sigmaX, gamma, X0, U0, deltaT, H, N,
                 rng=np.random.default_rng()):
        super().__init__(nParticles, resampleLimit, rng=rng)
        self.generator = ProjectModel(muU, muX, sigmaX, gamma, X0, U0)
        self.particles = [ProjectParticle(self.generator, deltaT, H, N) for _ in range(nParticles)]

    @staticmethod
    def particle_increment(particle, observation, index, deltaT):
        return particle.weight_increment(observation=observation, index=index, deltaT=deltaT)

    def increment_particles(self, obs, index, deltaT):
        weightIncrements = t_map(partial(self.particle_increment, observation=obs, index=index, deltaT=deltaT),
                                 self.particles,
                                 disable=True)
        return self.logWghts + weightIncrements

    def particle_MCMC(self, particle, rho=0.254):
        mcmc = pCnMCMC(self.compute_likelihood, rng=self.rng) # TODO: Check how to pass the current X to the likelihood
        curr = particle.trajectory
        newTraj = mcmc.proposal(curr, rho=rho)  # TODO: Check particle.trajectory correct dimensions
        particle.trajectory = mcmc.accept_reject(newTraj, curr)  # TODO: CHeck we dont need to change any other property

    def MCMC_step(self, rho=0.254):
        t_map(partial(self.particle_MCMC, rho=rho),self.particles,disable=True)

    def run_filter(self, observation, index, deltaT, rho=0.254):
        self.logWghts, self.particles = self.resample()  # Resampling
        self.MCMC_step(rho=rho)  # MCMC step
        self.logWghts = self.increment_particles(obs=observation, index=index,
                                                 deltaT=deltaT)  # Sample proposal and reweight
        self.logWghts = self.normalise_weights()


class ProjectParticle(Particle):
    def __init__(self, model, deltaT, H, N):
        super().__init__()
        self.generator = model
        self.noiseIncrements = self.generator.sample_increments(deltaT=deltaT, H=H, N=N)  # N fBn samples, Step 3
        self.trajectory = np.array([model.get_initial_state()])  # TODO: check consistency of dimensions across modules

    def state_increment(self, prev, deltaT, noiseIncrement):
        return self.generator.inverse_lamperti(
            self.generator.increment_filter(prev=prev, deltaT=deltaT, M=noiseIncrement))

    def weight_increment(self, observation, deltaT, index):
        m = observation + (self.generator.obsMean - 0.5 * np.exp(self.trajectory[-1] / 2.)) * deltaT
        sigma = np.exp(self.trajectory[-1]) * deltaT
        self.trajectory=np.append(self.trajectory,
            self.state_increment(prev=self.trajectory[-1], deltaT=deltaT, noiseIncrement=self.noiseIncrements[index]))
        return np.log(snorm.pdf(observation, loc=m, scale=np.sqrt(sigma)))


def test():
    muU = 0.1
    muX = 0.1
    gamma = 1.
    sigmaX = np.sqrt(muX * gamma / 0.55)
    X0 = 1.
    U0 = 0.
    deltaT = 0.001
    H = 0.8
    N = 2 * 11
    m = ProjectModel(muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    L = Us.size
    pf = ProjectParticleFilter(nParticles=500, resampleLimit=0., muU=muU, muX=muX, sigmaX=sigmaX, gamma=gamma, X0=X0,
                               U0=U0, deltaT=deltaT, H=H, N=N)
    for j in range(1, L):
        pf.run_filter(observation=Us[j], index=j, deltaT=deltaT)


test()
