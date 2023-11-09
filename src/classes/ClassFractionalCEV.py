from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalCEV:

    def __init__(self, muU: float, alpha: float, sigmaX: float, muX: float, X0: float, U0: float,
                 rng: np.random.Generator = np.random.default_rng()):
        assert (alpha > 0.)  # For mean-reversion and not exponential explosion
        assert (X0 > 0.)  # Initial vol cannot be 0
        self.obsMean = muU  # Log price process mean
        self.volMean = muX
        self.volVol = sigmaX
        self.stanMeanRev = alpha
        self.initialVol = X0
        self.initialLogPrice = U0
        self.rng = rng
        self.gaussIncs = None

    def get_initial_state(self):
        return self.initialVol

    def sample_increments(self, H: float, N: int, gaussRvs: np.ndarray) -> np.ndarray:
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = fBn.circulant_simulation(N_samples=N, gaussRvs=gaussRvs)  # Scale over timescale included in circulant
        return incs

    def lamperti(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return np.power(self.volVol, -1) * np.log(x / self.initialVol)

    def inverse_lamperti(self, Z: np.ndarray):
        return self.initialVol * np.exp(self.volVol * Z)

    def increment_simulation(self, prev: np.ndarray, currX: np.ndarray, deltaT: float):
        """ Increment log prices """
        driftU = self.obsMean - 0.5 * np.exp(currX)
        stdU = np.sqrt(deltaT) * np.exp(currX / 2.)
        return prev + driftU * deltaT + stdU * self.rng.normal()

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        """ Increment volatilities """
        driftZ = self.stanMeanRev * self.volMean * np.power(self.initialVol, -1) * np.exp(-self.volVol * prev)
        driftZ += -0.5 * self.volVol - self.stanMeanRev
        return prev + driftZ * deltaT + M

    def observation_mean(self, prevObs: np.ndarray, currX: np.ndarray, deltaT: float):
        return prevObs + (self.obsMean - 0.5 * np.exp(currX)) * deltaT  # U_i-1 +(muU-0.5exp(X))delta

    @staticmethod
    def observation_var(currX: np.ndarray, deltaT: float):
        # return deltaT
        return np.exp(currX) * deltaT  # delta exp(currX)

    def state_simulation(self, H: float, N: int, deltaT: float, X0: float = None, Ms: np.ndarray = None,
                         gaussRvs: np.ndarray = None):
        if X0 is None:
            Zs = [self.lamperti(self.initialVol)]
        else:
            Zs = [self.lamperti(X0)]
        if gaussRvs is None:
            self.gaussIncs = self.rng.normal(size=2 * N)
        else:
            self.gaussIncs = gaussRvs
        if Ms is None:
            Ms = self.sample_increments(H=H, N=N, gaussRvs=self.gaussIncs)
        for i in range(1, N + 1):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))
        return self.inverse_lamperti(np.array(Zs))

    def euler_simulation(self, H: float, N: int, deltaT: float, Ms: np.ndarray = None, gaussRvs: np.ndarray = None):
        assert (0. < H < 1. and deltaT == 1. / N)  # Ensure we simulate from 0 to 1
        Zs = [self.lamperti(self.initialVol)]
        Us = [self.initialLogPrice]
        if gaussRvs is None:
            self.gaussIncs = self.rng.normal(size=2 * N)
        else:
            self.gaussIncs = gaussRvs
        if Ms is None:
            Ms = self.sample_increments(H=H, N=N, gaussRvs=self.gaussIncs)
        for i in range(1, N + 1):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))  # Ms[0] = B^H_1 - B^H_0
            Us.append(self.increment_simulation(prev=Us[i - 1], currX=self.inverse_lamperti(Zs[i]), deltaT=deltaT))
        """ Use inverse Lamperti transform to return volatilities """
        return self.inverse_lamperti(np.array(Zs)), np.array(Us)
