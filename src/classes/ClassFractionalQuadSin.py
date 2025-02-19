from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from tqdm import tqdm

class FractionalQuadSin:

    def __init__(self, sin_space_scale: float, quad_coeff: float, sin_coeff: float, diff: float, X0: float = 0,
                 rng: np.random.Generator = np.random.default_rng()):
        self.sin_space_scale = sin_space_scale
        self.quad_coeff = quad_coeff
        self.sin_coeff = sin_coeff
        self.diff = diff
        self.initialVol = X0
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

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        # driftX = -V'(x) where V'(x) = -(ax^2-b*cos(cx)) potential --> 2ax+bcsin(cx) drift
        driftX = 2.*self.quad_coeff * prev + self.sin_coeff * self.sin_space_scale*np.sin(self.sin_space_scale*prev)  # gamma*sin(Xt-1)
        diffX = self.diff * M
        return prev + driftX * deltaT + diffX

    def euler_simulation(self, H: float, N: int, isUnitInterval: bool, t0: float, t1: float, deltaT: float,
                         X0: float = None, Ms: np.ndarray = None,
                         gaussRvs: np.ndarray = None):
        time_ax = np.arange(start=t0, stop=t1 + deltaT, step=deltaT)
        assert (time_ax[-1] == t1 and time_ax[0] == t0)
        if X0 is None:
            Zs = [self.initialVol]  # [self.lamperti(self.initialVol)]
        else:
            Zs = [X0]  # [self.lamperti(X0)]
        Ds = [0.]
        if gaussRvs is None:
            if H != 0.5:
                self.gaussIncs = self.rng.normal(size=2 * N)
            else:
                self.gaussIncs = self.rng.normal(size=N)
        else:
            self.gaussIncs = gaussRvs
        if Ms is None:
            if H != 0.5:
                Ms = self.sample_increments(H=H, N=N, gaussRvs=self.gaussIncs, isUnitInterval=isUnitInterval)
            else:
                Ms = self.gaussIncs * np.sqrt(deltaT)
        for i in (range(1, N + 1)):
            Z = self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1])
            Zs.append(Z)
        return np.array(Zs)
