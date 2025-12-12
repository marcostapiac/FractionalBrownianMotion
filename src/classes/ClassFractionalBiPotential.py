from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalBiPotential:

    def __init__(self, num_dims:int, const: float, quartic_coeff: float, quad_coeff: float, diff: float, X0,
                 rng: np.random.Generator = np.random.default_rng()):
        assert (num_dims >= 1)
        self.ndims = num_dims

        if self.ndims == 1:
            self.const = const
            self.quartic_coeff = quartic_coeff
            self.quad_coeff = quad_coeff
            self.initialVol = X0
        else:
            self.const = np.array(const)
            self.quartic_coeff = np.array(quartic_coeff)
            self.quad_coeff = np.array(quad_coeff)
            self.initialVol = np.array(X0)
        self.diff = diff
        self.rng = rng
        self.gaussIncs = None

    def get_initial_state(self):
        return self.initialVol

    def sample_increments(self, H: float, N: int, isUnitInterval: bool, gaussRvs: np.ndarray) -> np.ndarray:
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = fBn.circulant_simulation(N_samples=N, gaussRvs=gaussRvs,
                                        scaleUnitInterval=isUnitInterval)  # Scale over timescale included in circulant
        return incs

    @staticmethod
    def lamperti(self, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return None  # np.power(self.volVol, -1) * np.log(x / self.initialVol)

    @staticmethod
    def inverse_lamperti(self, Z: np.ndarray):
        return None  # self.initialVol * np.exp(self.volVol * Z)

    def increment_state(self, prev: np.ndarray, deltaT: float, M: np.ndarray):
        # driftX = -V'(x) where V(x) = ax^4+bx^2+cx
        #assert prev.shape[-1] == M.shape[-1] and M.shape[-1] != 1
        driftX = -(4.*self.quartic_coeff * np.power(prev, 3) + 2.*self.quad_coeff * prev + self.const)
        diffX = self.diff * M[np.newaxis, :]
        ## See (Weak approximation schemes for SDEs with super-linearly growing coefficients, 2023) for weak solution
        #diffX = diffX/(1.+deltaT*driftX)
        # See Tamed Euler
        if self.ndims > 1:
            driftX = driftX/(1.+deltaT*np.abs(driftX))
        return prev + driftX * deltaT + diffX

    def euler_simulation(self, H: float, N: int, isUnitInterval: bool, t0: float, t1: float, deltaT: float,
                         X0 = None, Ms: np.ndarray = None,
                         gaussRvs: np.ndarray = None):
        time_ax = np.arange(start=t0, stop=t1 + deltaT, step=deltaT)
        if t1==1: assert (isUnitInterval == True)
        assert (time_ax[-1] == t1 and time_ax[0] == t0)
        if X0 is None:
            Zs = [self.initialVol]  # [self.lamperti(self.initialVol)]
        else:
            Zs = [X0]  # [self.lamperti(X0)]
        if gaussRvs is None:
            if H != 0.5:
                self.gaussIncs = self.rng.normal(size=2 * N)
            else:
                self.gaussIncs = self.rng.normal(size=(N, self.ndims))
        else:
            self.gaussIncs = gaussRvs
        if Ms is None:
            if H != 0.5:
                Ms = self.sample_increments(H=H, N=N, gaussRvs=self.gaussIncs, isUnitInterval=isUnitInterval)
            else:
                Ms = self.gaussIncs * np.sqrt(deltaT)
        for i in (range(1, N + 1)):
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1,:]))
        print(len(Zs))
        print(Zs)
        return np.array(Zs)
