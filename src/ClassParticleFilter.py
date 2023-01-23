from utils.math_functions import np, logsumexp, snorm
from p_tqdm import t_map
from functools import partial
from copy import deepcopy


class Particle:
    def __init__(self, rng=np.random.default_rng()):
        self.rng = rng

    def state_increment(self, **kwargs):
        pass

    def weight_increment(self, *kwargs):
        pass


class ParticleFilter:
    def __init__(self, nParticles, rng=np.random.default_rng()):
        self.M = nParticles
        self.logWghts = np.array([-np.log(nParticles) for _ in range(nParticles)]).reshape(
            (1, nParticles))  # Incremental weight updates since using resampling
        self.particles = [Particle() for _ in range(nParticles)]
        self.rng = rng

    def get_log_Neff(self):
        return -logsumexp(2. * self.logWghts, lambda x: 1., np.ones(self.M), isLog=True)

    def compute_incremental_likelihood(self, isLog=False):
        return logsumexp(self.logWghts, lambda x: 1., np.ones(self.M), isLog=isLog)

    def normalise_weights(self):
        lsum_weights = logsumexp(self.logWghts, lambda x: 1., np.ones(self.M), isLog=True)
        return self.logWghts - lsum_weights

    def resample(self):
        """ Adapted from filterpy.monte_carlo.stratified_resample """
        nP = self.M
        new_ps = [list([]) for _ in range(nP)]
        """
        u = np.zeros((nP, 1))
        c = np.cumsum(np.exp(self.logWghts))
        c[-1] = 1.0
        i = 0
        u[0] = self.rng.random() / nP
        resampled_indices = []
        for j in range(nP):

            u[j] = u[0] + j / nP

            while u[j] > c[i]:
                i = i + 1

            new_ps[j] = deepcopy(self.particles[i])
            resampled_indices.append(i)
        # print(np.unique(resampled_indices).shape[0] / nP)  # Check how many unique particles are propagated fwds
        """
        indxs = self.rng.choice(nP, nP, p=np.squeeze(np.exp(self.logWghts)))
        for i, indx in zip(np.arange(0, nP, step=1), indxs):
            new_ps[i] = deepcopy(self.particles[indx])
        logWeights = np.array([-np.log(nP)] * nP)
        return np.atleast_2d(logWeights), new_ps


class FractionalParticleFilter(ParticleFilter):
    def __init__(self, nParticles, model, deltaT, H, N,
                 rng=np.random.default_rng()):
        super().__init__(nParticles, rng=rng)
        self.generator = model
        self.particles = [ProjectParticle(model=self.generator, deltaT=deltaT, H=H, N=N, rng=rng) for _ in range(nParticles)]
        self.logLEstimate = self.compute_incremental_likelihood(isLog=True)

    @staticmethod
    def particle_increment(particle, observation, deltaT, index, vol=None):
        return particle.weight_increment(observation=observation, deltaT=deltaT, index=index, vol=vol)

    def increment_particles(self, obs, deltaT, index, vol=None):
        weightIncrements = t_map(partial(self.particle_increment, observation=obs, deltaT=deltaT, index=index, vol=vol),
                                 self.particles,
                                 disable=True)
        return weightIncrements

    def particle_move(self, particle, deltaT, rho=0.1):
        particle.gaussRvs = np.sqrt(1. - np.power(rho, 2)) * particle.gaussRvs + rho * self.rng.normal(
            size=len(particle.gaussRvs))  # pCn
        particle.fBmIncrements = particle.generator.sample_increments(deltaT=deltaT, H=particle.H, N=particle.N,
                                                                      gaussRvs=particle.gaussRvs)  # N fBn samples, Step 3
        L = len(particle.trajectory)
        particle.trajectory = particle.generator.state_simulation(H=particle.H, N=L - 1, deltaT=deltaT,
                                                                  Ms=particle.fBmIncrements[:L])

    def move_after_resample(self, deltaT, rho=0.014):
        t_map(partial(self.particle_move, deltaT=deltaT, rho=rho), self.particles, disable=True)

    def run_filter(self, observation, deltaT, index, vol=None):
        self.logWghts = self.increment_particles(obs=observation, deltaT=deltaT,
                                                 vol=vol, index=index)  # Sample proposal and reweight
        self.logLEstimate += self.compute_incremental_likelihood(isLog=True)
        self.logWghts = self.normalise_weights()
        # Normalised weights before resampling target p(y_i|y_1:i-1)
        self.logWghts, self.particles = self.resample()  # Resampling

    @staticmethod
    def particle_state_mean(particle):
        return particle.state_smoothing()

    @staticmethod
    def particle_obs_mean(particle):
        return particle.m

    @staticmethod
    def particle_obs_var(particle):
        return particle.sigma2

    def get_obs_mean_posterior(self):
        assert ((self.logWghts == -np.log(self.M)).all())  # Ensure calculation computed post-resampling
        ms = t_map(partial(self.particle_obs_mean), self.particles, disable=True)
        return logsumexp(np.atleast_2d(self.logWghts), lambda x: x, np.array(ms), axis=0, isLog=False)

    def get_vol_mean_posterior(self, vol=None):
        assert ((self.logWghts == -np.log(self.M)).all())  # Ensure normalised weights are being used
        ms = t_map(partial(self.particle_state_mean), self.particles, disable=True)
        return logsumexp(np.atleast_2d(self.logWghts), lambda x: x, np.array(ms), axis=0, isLog=False)


class ProjectParticle(Particle):
    def __init__(self, model, deltaT, H, N, rng=np.random.default_rng()):
        super().__init__(rng=rng)
        self.generator = model
        self.H = H
        self.N = N
        self.sigma2 = 1.
        self.m = 0.
        self.prevObs = model.initialLogPrice
        self.gaussRvs = self.rng.normal(size=2 * self.N)
        self.fBmIncrements = self.generator.sample_increments(deltaT=deltaT, H=H, N=N,
                                                              gaussRvs=self.gaussRvs)  # N fBn samples, Step 3
        self.trajectory = np.array([self.generator.get_initial_state()])

    def state_increment(self, prevVol, deltaT, fBmIncrement):
        return self.generator.inverse_lamperti(
            self.generator.increment_state(prev=self.generator.lamperti(prevVol), deltaT=deltaT, M=fBmIncrement))

    def state_smoothing(self):
        return self.trajectory[-1]

    def compute_obs_conditional_mean(self, prevObs, deltaT):
        return self.generator.observation_mean(prevObs, currX=self.trajectory[-1], deltaT=deltaT)

    def compute_obs_conditional_var(self, deltaT):
        return self.generator.observation_var(currX=self.trajectory[-1], deltaT=deltaT)

    def weight_increment(self, observation, deltaT, index, vol=None):
        if vol:
            X = self.state_increment(prevVol=vol[0], deltaT=deltaT, fBmIncrement=self.fBmIncrements[index - 1])
        else:
            X = self.state_increment(prevVol=self.trajectory[-1], deltaT=deltaT,
                                     fBmIncrement=self.fBmIncrements[index - 1])
        self.trajectory = np.append(self.trajectory, X)
        self.m = self.compute_obs_conditional_mean(prevObs=self.prevObs, deltaT=deltaT)
        self.sigma2 = self.compute_obs_conditional_var(deltaT=deltaT)
        self.prevObs = observation
        return snorm.logpdf(observation, loc=self.m, scale=np.sqrt(self.sigma2))
