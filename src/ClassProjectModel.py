from utils.math_functions import np
from src.ClassFractionalBrownianNoise import FractionalBrownianNoise


class ProjectModel:

    def __init__(self, muU, muX, sigmaX, gamma, X0, U0):
        assert (2. * muX * gamma / np.power(sigmaX, 2) - 0.5 > .5)  # Ensures positivity of vol process
        assert (gamma > 0.)  # For mean-reversion and not exponential explosion
        assert (X0 > 0.)  # Initial vol cannot be 0
        self.obsMean = muU  # Log price process mean
        self.volMean = muX
        self.volVol = sigmaX
        self.meanRev = gamma
        self.beta = 2. * self.volMean * self.meanRev / np.power(self.volVol,
                                                                2) - 0.5  # TODO: Why does Supp Material not include -0.5
        self.initialVol = X0  # TODO: How to set initial vol
        self.initialLogPrice = U0  # TODO: How to set initial log price

    def get_initial_state(self):
        return self.initialVol
    @staticmethod
    def sample_increments(deltaT, H, N):
        fBn = FractionalBrownianNoise(H=H)
        return (deltaT ** H) * fBn.davies_and_harte_simulation(N_samples=N)  # TODO: Check simulation is within [0, T]

    def increment_filter(self, prev, deltaT, M):
        """ Increment volatilities """
        driftZ = -0.5 * self.meanRev * prev + self.beta / prev
        return prev + driftZ * deltaT + M

    def lamperti(self, x):
        return 2. / self.volVol * np.sqrt(x)

    def inverse_lamperti(self, Z):
        return np.power((self.volVol / 2.) * Z, 2)

    def increment_simulation(self, prev, currZ, deltaT):
        """ Increment log prices """
        driftU = self.obsMean - 0.5 * np.exp(currZ)
        stdU = np.exp(currZ / 2.)
        return prev + driftU * deltaT + stdU * np.random.normal()

    def euler_simulation(self, H, N, deltaT):
        """ Full time Euler-Maruyama Simulation of Heston Vol Model """
        Zs = [self.lamperti(self.initialVol)]
        Us = [self.initialLogPrice]
        Ms = self.sample_increments(deltaT=deltaT, H=H, N=N)
        for i in range(1, N):
            Zs.append(self.increment_filter(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i]))  # Ms[0] = B^H_1 - B^H_0
            Us.append(self.increment_simulation(prev=Us[i - 1], currZ=Zs[i], deltaT=deltaT))
        """ Use inverse Lamperti transform to return volatilities """
        return self.inverse_lamperti(np.array(Zs)), np.array(Us)

