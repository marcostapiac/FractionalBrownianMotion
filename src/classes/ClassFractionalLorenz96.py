from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from tqdm import tqdm


class FractionalLorenz96:

    def __init__(self, forcing_const: float, diff: float, num_dims: int, X0: np.ndarray,
                 rng: np.random.Generator = np.random.default_rng()):
        assert (num_dims >= 4)
        self.forcing_const = forcing_const
        self.ndims = num_dims
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

    def drift_X(self, prev):
        assert (len(prev.shape) == 1 and prev.shape[0] == self.ndims)
        driftX = np.zeros_like(prev)
        for i in range(self.ndims):
            driftX[i] = (prev[(i + 1) % self.ndims] - prev[i - 2]) * prev[i - 1] - prev[i] + self.forcing_const
        return driftX

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        # driftX = -V'(x) where V'(x) = -(ax^2-b*cos(cx)) potential --> 2ax+bcsin(cx) drift
        driftX = self.drift_X(prev=prev)
        diffX = self.diff * M
        return prev + driftX * deltaT + diffX

    def euler_simulation(self, H: float, N: int, isUnitInterval: bool, t0: float, t1: float, deltaT: float,
                         X0: np.ndarray, Ms: np.ndarray = None,
                         gaussRvs: np.ndarray = None):
        # assert ((isUnitInterval and t0 == 0. and t1 == 1.) or ((not isUnitInterval) and t0 == 0. and t1 != 1.))
        time_ax = np.arange(start=t0, stop=t1 + deltaT, step=deltaT)
        assert (time_ax[-1] == t1 and time_ax[0] == t0)
        if X0 is None:
            Zs = [self.initialVol]  # [self.lamperti(self.initialVol)]
        else:
            Zs = [X0]  # [self.lamperti(X0)]
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
            Zs.append(self.increment_state(prev=Zs[i - 1], deltaT=deltaT, M=Ms[i - 1]))
        return np.array(Zs)
