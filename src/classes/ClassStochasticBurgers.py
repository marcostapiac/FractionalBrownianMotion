from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise


class StochasticBurgers:

    def __init__(self, num_dims:int,  diff: float, nu, num_modes, alpha, X0,
                 rng: np.random.Generator = np.random.default_rng()):
        assert (num_dims >= 1)
        self.ndims = num_dims
        self.nu = nu
        self.num_fourier_modes = num_modes
        self.alpha = alpha
        self.L = 2 * np.pi
        self.kappa = 2.0 * np.pi / self.L
        self.diff = diff
        self.gaussIncs = None
        self.initialVol = X0
        self.rng = np.random.default_rng()

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


    def build_q_nonneg(self, m):
        """
        Build per-mode variances q_m for m=0..K (nonnegative), with sum(q_m)=sigma_tot^2.
        Monotone decreasing in |k| for powerlaw/gaussian.
        """
        k = self.kappa * m.astype(float)
        q = np.zeros_like(k, dtype=float)
        if q.shape[0] > 1:
            q[1:] = 2 * self.nu * 5 * (np.abs(k[1:]) ** (-self.alpha))
        return q

    # ---------------------------- convolution helper -------------------------- #
    def nonlinear_term_convolution(self, a_nonneg):
        """
        Compute Nh for nonnegative modes using explicit convolution on truncated band.
        a_nonneg: complex array length M = K+1 for m = 0..K (nonnegative).
        Returns Nh_nonneg: complex array length M corresponding to modes m = 0..K.
        """
        K = self.num_fourier_modes - 1
        # Build full symmetric spectrum a_full for m=-K..K (index shift by +K)
        a_full = np.empty(2 * K + 1, dtype=complex)
        a_full[K] = a_nonneg[0]  # m=0 at index K
        for m in range(1, K + 1):
            a_full[K + m] = a_nonneg[m]  # +m
            a_full[K - m] = np.conj(a_nonneg[m])  # -m (conjugate)
        # Convolution: (u^2)_m = sum_{p+q=m} a_p a_q with |p|,|q|≤K
        Nh_nonneg = np.zeros(self.num_fourier_modes, dtype=complex)
        for m in range(0, K + 1):  # target mode m ≥ 0
            if m == 0:
                Nh_nonneg[m] = 0.0  # derivative kills mean
                continue
            ik = 1j * self.kappa * m
            s = 0.0 + 0.0j
            for p in range(-K, K + 1):
                q = m - p
                if -K <= q <= K:
                    ap = a_full[K + p]
                    aq = a_full[K + q]
                    s += ap * aq
            Nh_nonneg[m] = -0.5 * ik * s
        return Nh_nonneg

    def increment_state(self, prev: np.ndarray, deltaT, k_phys, q):

        Nh_now = self.nonlinear_term_convolution(prev)  # complex (M,)

        # Spectral noise with E|eta_m|^2 = q_m (real at m=0, complex for m>=1)
        eta = np.zeros(self.num_fourier_modes, dtype=complex)
        eta[0] = np.sqrt(q[0]) * self.rng.standard_normal()
        if self.num_fourier_modes > 1:
            re = self.rng.standard_normal(self.num_fourier_modes - 1)
            im = self.rng.standard_normal(self.num_fourier_modes - 1)
            eta[1:] = np.sqrt(q[1:] / 2.0) * (re + 1j * im)

        # Exponential Euler update:
        # a_{n+1} = E * a_n + B * N(a_n) + F * eta   (elementwise over modes)
        denom = 1.0 + self.nu * (k_phys ** 2) * deltaT

        prev = (prev + deltaT * Nh_now + np.sqrt(deltaT) * eta) / denom

        return prev

    def euler_simulation(self, N: int, isUnitInterval: bool, t0: float, t1: float, deltaT: float,
                         X0 = None):
        time_ax = np.arange(start=t0, stop=t1 + deltaT, step=deltaT)
        if t1==1: assert (isUnitInterval == True)
        assert (time_ax[-1] == t1 and time_ax[0] == t0)
        Reals = [self.initialVol.real]  # [self.lamperti(self.initialVol)]
        Imags = [self.initialVol.imag]  # [self.lamperti(self.initialVol)]
        m = np.arange(self.num_fourier_modes, dtype=int)  # 0..K
        k_phys = self.kappa * m.astype(float)
        q = self.build_q_nonneg(m)
        prev = self.initialVol
        for _ in (range(1, N + 1)):
            new = self.increment_state(prev=prev, q=q, deltaT=deltaT, k_phys=k_phys)
            Reals.append(prev.real)
            Imags.append(prev.imag)
            prev = new
        Reals = np.concatenate(Reals, axis=0).reshape((N+1, self.num_fourier_modes, 1))
        Imags = np.concatenate(Imags, axis=0).reshape((N+1, self.num_fourier_modes, 1))
        return np.concatenate([Reals, Imags], axis=-1).reshape((N+1, self.num_fourier_modes, 2))
