
import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise


class FractionalMullerBrown:

    def __init__(self, diff: float, initialState: np.ndarray,X0s:np.ndarray, Y0s:np.ndarray, Aks:np.ndarray, aks:np.ndarray, bks:np.ndarray, cks:np.ndarray, 
                 rng: np.random.Generator = np.random.default_rng()):
        self.ndims = 2
        self.diff = diff
        self.Aks = Aks
        self.aks = aks
        self.bks = bks
        self.cks = cks
        assert (self.Aks.shape[0] == self.aks.shape[0] == self.bks.shape[0] == self.cks.shape[0] == 4)
        self.initalVol = initialState
        self.X0 = X0s
        self.Y0 = Y0s
        self.rng = rng
        self.gaussIncs = None

    def get_initial_state(self):
        return self.initalVol

    def sample_increments(self, H: float, N: int, isUnitInterval: bool, gaussRvs: np.ndarray) -> np.ndarray:
        fBn = FractionalBrownianNoise(H=H, rng=self.rng)
        incs = fBn.circulant_simulation(N_samples=N, gaussRvs=gaussRvs,
                                        scaleUnitInterval=isUnitInterval)  # Scale over timescale included in circulant
        return incs

    def drift_X(self, prev, deltaT):
        assert (len(prev.shape) == 1 and prev.shape[0] == self.ndims)
        common = self.Aks * np.exp(self.aks*np.power(prev[0]-self.X0,2) \
                 +self.bks*(prev[0]-self.X0)*(prev[1]-self.Y0)
                + self.cks*np.power(prev[1] - self.Y0,2))
        assert (common.shape[0] == 4)
        driftX = np.zeros(shape=self.ndims)
        driftX[0] = -np.sum(common*(2.*self.aks*(prev[0]-self.X0) + self.bks*(prev[1]-self.Y0)))
        driftX[1] = -np.sum(common*(2.*self.cks*(prev[1]-self.Y0) + self.bks*(prev[0]-self.X0)))
        return driftX

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        driftX = self.drift_X(prev=prev, deltaT=deltaT)
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
