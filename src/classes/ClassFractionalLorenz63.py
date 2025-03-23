
import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalLorenz63:

    def __init__(self, diff: float, sigma: float, beta: float, rho: float, initialState: np.ndarray, rng: np.random.Generator = np.random.default_rng()):
        self.ndims = 3
        self.diff = diff
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.initalVol = initialState
        self.rng = rng
        self.gaussIncs = None

    def get_initial_state(self):
        return self.initalVol

    def sample_increments(self, H: float, N: int, isUnitInterval: bool, gaussRvs: np.ndarray) -> np.ndarray:
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = fBn.circulant_simulation(N_samples=N, gaussRvs=gaussRvs,
                                        scaleUnitInterval=isUnitInterval)  # Scale over timescale included in circulant
        return incs

    def drift_X(self, prev):
        driftX = np.zeros(self.ndims)
        driftX[0] = self.sigma * (prev[1] - prev[0])
        driftX[1] = (prev[0] * (self.rho - prev[2]) - prev[1])
        driftX[2] = (prev[0] * prev[1] - self.beta * prev[2])
        return driftX

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        driftX = self.drift_X(prev=prev)
        diffX = self.diff * M
        return prev + driftX * deltaT + diffX

    def euler_simulation(self, H: float, N: int, t0: float, t1: float, deltaT: float, X0: np.ndarray,
                         Ms: np.ndarray = None, gaussRvs: np.ndarray = None):
        time_ax = np.arange(start=t0, stop=t1 + deltaT, step=deltaT)
        isUnitInterval = True if np.allclose(t1, 1.) else False
        assert (time_ax[-1] == t1 and time_ax[0] == t0)
        if X0 is None:
            Zs = [self.initalVol]  # [self.lamperti(self.X0)]
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
