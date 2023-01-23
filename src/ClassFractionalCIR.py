from utils.math_functions import np
from src.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalCIR:

    def __init__(self, muU, muX, sigmaX, gamma, X0, U0, rng=np.random.default_rng()):
        assert (2. * muX * gamma / np.power(sigmaX, 2) - 0.5 > .5)  # Ensures positivity of vol process
        assert (gamma > 0.)  # For mean-reversion and not exponential explosion
        assert (X0 > 0.)  # Initial vol cannot be 0
        self.obsMean = muU  # Log price process mean
        self.volMean = muX
        self.volVol = sigmaX
        self.meanRev = gamma
        self.beta = (2. * self.volMean * self.meanRev / np.power(self.volVol, 2)) - 0.5
        self.initialVol = X0
        self.initialLogPrice = X0 + rng.normal()#U0
        self.rng = rng

    def get_initial_state(self):
        return self.initialVol

    def sample_increments(self, deltaT, H, N, gaussRvs=None):
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = (deltaT ** H) * fBn.davies_and_harte_simulation(N_samples=N, gaussRvs=gaussRvs)
        return incs

    def lamperti(self, x):
        return (2. / self.volVol) * np.sqrt(x)

    def inverse_lamperti(self, Z):
        return np.power((self.volVol / 2.) * Z, 2)

    def increment_simulation(self, prev, currX, deltaT):
        """ Increment log prices """
        driftU = self.obsMean - 0.5 * np.exp(currX)
        stdU = np.sqrt(deltaT) * np.exp(currX / 2.)
        #return currX + np.sqrt(deltaT)*self.rng.normal()
        return prev + driftU * deltaT + stdU * self.rng.normal()

    def increment_state(self, prev, deltaT, M):
        """ Increment volatilities """
        driftZ = -0.5 * self.meanRev * prev + self.beta / prev
        return prev + driftZ * deltaT + M

    def observation_mean(self, prevObs, currX, deltaT):
        #return currX
        return prevObs + (self.obsMean - 0.5 * np.exp(currX)) * deltaT

    def observation_var(self, currX, deltaT):
        #return deltaT
        return np.exp(currX)*deltaT

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
        """ Full time Euler-Maruyama Simulation of Heston Vol Model """
        Zs = [self.lamperti(self.initialVol)]
        Us = [self.initialLogPrice]
        if Ms is None:
            Ms = self.sample_increments(deltaT=deltaT, H=H, N=N, gaussRvs=gaussRvs)
        for i in range(1, N + 1):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))  # Ms[0] = B^H_1 - B^H_0
            Us.append(self.increment_simulation(prev=Us[i - 1], currX=self.inverse_lamperti(Zs[i]), deltaT=deltaT))
        """ Use inverse Lamperti transform to return volatilities """
        return self.inverse_lamperti(np.array(Zs)), np.array(Us)
