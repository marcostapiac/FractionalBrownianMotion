from utils.math_functions import np
from src.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalCEV:

    def __init__(self, muU, muX, sigmaX, gamma, X0, U0, rng=np.random.default_rng()):
        assert (gamma > 0.)  # For mean-reversion and not exponential explosion
        assert (X0 > 0.)  # Initial vol cannot be 0
        self.obsMean = muU  # Log price process mean
        self.volMean = muX
        self.volVol = sigmaX
        self.meanRev = gamma
        self.initialVol = X0
        self.initialLogPrice = U0
        self.rng = rng

    def get_initial_state(self):
        return self.initialVol

    def sample_increments(self, deltaT, H, N, gaussRvs=None):
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = (deltaT ** H) * fBn.davies_and_harte_simulation(N_samples=N, gaussRvs=gaussRvs)
        return incs

    def lamperti(self, x):
        return np.power(self.volVol,-1) * np.log(x / self.initialVol)

    def inverse_lamperti(self, Z):
        return self.initialVol * np.exp(self.volVol * Z)

    def increment_simulation(self, prev, currX, deltaT):
        """ Increment log prices """
        driftU = self.obsMean - 0.5 * np.exp(currX)
        stdU = np.sqrt(deltaT) * np.exp(currX / 2.)
        # return currX + np.sqrt(deltaT)*self.rng.normal()
        return prev + driftU * deltaT + stdU * self.rng.normal()

    def increment_state(self, prev, deltaT, M):
        """ Increment volatilities """
        driftZ = self.meanRev * self.volMean * np.power(self.volVol * self.initialVol, -1) * np.exp(-self.volVol * prev)
        driftZ += -0.5 * self.volVol - self.meanRev * np.power(self.volVol, -1)
        return prev + driftZ * deltaT + M

    def observation_mean(self, prevObs, currX, deltaT):
        # return currX
        return prevObs + (self.obsMean - 0.5 * np.exp(currX)) * deltaT # U_i-1 +(muU-0.5exp(X))delta

    def observation_var(self, currX, deltaT):
        # return deltaT
        return np.exp(currX) * deltaT # delta exp(currX)

    def state_simulation(self, H, N, deltaT, X0=None, Ms=None, gaussRvs=None):
        if X0 is None:
            Zs = [self.lamperti(self.initialVol)]
        else:
            Zs = [self.lamperti(X0)]
        if Ms is None:
            Ms = self.sample_increments(deltaT=deltaT, H=H, N=N, gaussRvs=gaussRvs)
        for i in range(1, N + 1):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))
        return self.inverse_lamperti(np.array(Zs))

    def euler_simulation(self, H, N, deltaT, Ms=None, gaussRvs=None):
        Zs = [self.lamperti(self.initialVol)]
        Us = [self.initialLogPrice]
        if Ms is None:
            Ms = self.sample_increments(deltaT=deltaT, H=H, N=N, gaussRvs=gaussRvs)
        for i in range(1, N + 1):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))  # Ms[0] = B^H_1 - B^H_0
            Us.append(self.increment_simulation(prev=Us[i - 1], currX=self.inverse_lamperti(Zs[i]), deltaT=deltaT))
        """ Use inverse Lamperti transform to return volatilities """
        return self.inverse_lamperti(np.array(Zs)), np.array(Us)
