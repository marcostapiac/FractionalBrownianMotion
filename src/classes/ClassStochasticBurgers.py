from typing import Union

import numpy as np

from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from tqdm import tqdm
from numpy.fft import rfft, irfft, rfftfreq

class StochasticBurgers:

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

    @staticmethod
    def build_spectrum(k, spectrum="powerlaw", sigma_tot=0.2, ell=0.5, alpha=1.0, kc_ratio=0.3, zero_mean=True):
        """
        Return per-mode variances q_k (nonnegative k for rFFT) with sum(q_k) = sigma_tot^2.
        Monotone decreasing with |k| for powerlaw/gaussian; trace-class by construction.

        Args:
          k : array of nonnegative wavenumbers (shape Nf = Nx//2+1)
          spectrum : 'powerlaw' | 'gaussian' | 'band'
          sigma_tot : sqrt of the total variance (Σ q_k) injected per time step (spectral)
          ell : correlation length parameter (controls decay rate vs k)
          alpha : powerlaw smoothness (must be > 0.5 in 1D for trace-class in the continuum)
          kc_ratio : for 'band', cutoff as fraction of Nyquist (0..1]
          zero_mean : if True, set q_0 = 0

        Returns:
          q : array of shape like k with Σ q = sigma_tot^2
        """
        if spectrum == "powerlaw":
            shape = (1.0 + (ell * k) ** 2) ** (-alpha)
        elif spectrum == "gaussian":
            shape = np.exp(-(ell * k) ** 2)
        elif spectrum == "band":
            kmax = k.max()
            shape = (k <= kc_ratio * kmax).astype(float)
        else:
            raise ValueError("Unknown spectrum: choose 'powerlaw', 'gaussian', or 'band'.")

        if zero_mean:
            shape[0] = 0.0

        # Normalise to desired trace (sum of q_k)
        s = np.sum(shape)
        if s <= 0:
            raise ValueError(
                "Spectrum shape sums to zero; adjust parameters (e.g., increase kc_ratio or reduce zero_mean).")
        q = (sigma_tot ** 2) * shape / s
        return q

    def simulate_burgers_1d_once(self, deltaT:float,
            L=2 * np.pi, Nx=20, T=1.0, Nt=256,
            nu=0.05,
            spectrum="powerlaw", sigma_tot=0.2, ell=0.5, alpha=1.0, kc_ratio=0.3, zero_mean=True,
            rng=None
    ):
        """
            Single increment. Returns (U) with shape U: (1, Nx)
        """
        # grids
        dx = L / Nx
        x = np.linspace(0.0, L, Nx, endpoint=False)
        # spectral objects (nonnegative frequencies for rFFT)
        k = 2 * np.pi * rfftfreq(Nx, d=dx)  # shape Nf = Nx//2+1
        K2 = k * k
        k_max = k.max()

        # 2/3 de-aliasing mask for quadratic nonlinearity
        kmask = (k <= (2.0 / 3.0) * k_max).astype(float)

        # trace-class spatial covariance (per-mode variances q_k)
        qk = self.build_spectrum(k, spectrum=spectrum, sigma_tot=sigma_tot, ell=ell, alpha=alpha, kc_ratio=kc_ratio,
                                 zero_mean=zero_mean)

        # initial condition (smooth)
        uh = rfft(prev)

        # implicit diffusion denominator (per mode)
        denom = 1.0 + nu * K2 * deltaT

        # storage
        U = np.empty((1, Nx), dtype=float)
        t = np.linspace(0.0, T, Nt + 1)

        # time stepping
        for n in range(Nt):
        # Nonlinear term: -1/2 * ∂x(u^2)
        u = irfft(uh, n=Nx)
        qh = rfft(u * u)
        qh *= kmask  # de-alias after product
        Nh = -0.5j * k * qh

        # Spectral noise: complex Gaussian with per-mode variance qk
        nf = len(k)
        zeta = rng.standard_normal(nf) + 1j * rng.standard_normal(nf)
        # endpoints real for rFFT representation
        zeta[0] = rng.standard_normal()
        if Nx % 2 == 0:
            zeta[-1] = rng.standard_normal()

        etah = np.sqrt(qk) * zeta  # per-step spectral kick (variance qk)
        # Euler–Maruyama with implicit L
        uh = (uh + deltaT * Nh + np.sqrt(deltaT) * etah) / denom

        return irfft(uh, n=Nx)

    def increment_state(self, prev: np.ndarray, deltaT: float, M: int):
        # !/usr/bin/env python3
        """
        Stochastic Burgers 1D — I i.i.d. paths with trace-class spatial covariance.

        PDE: u_t + 1/2 (u^2)_x = nu u_xx + ξ(x,t),  periodic x in [0, L)

        Time:  [0, T] with Nt steps (Euler–Maruyama in time)
        Space: Nx grid points (pseudospectral: rFFT/irFFT, 2/3 de-aliasing)

        Noise model (white in time, colored in space):
          ξ(x,t) = sum_k sqrt(q_k) e^{ikx} dβ_k/dt,
        where q_k = spectrum(k) with Σ_k q_k = σ_tot^2 (trace-class). We provide:
          - powerlaw: q_k ∝ (1 + (ℓ|k|)^2)^(-α), α > 1/2 in 1D
          - gaussian: q_k ∝ exp(-(ℓ|k|)^2)
          - band:     q_k = const for |k| ≤ k_c, 0 otherwise
        We zero the mean mode by default (q_0 = 0) to avoid drifting the spatial average.

        Usage examples:
          python burgers_spde_1d_tracecov.py --I 10 --Nx 20 --Nt 256 --spectrum powerlaw --alpha 1.0 --ell 0.5 --sigma_tot 0.2
          python burgers_spde_1d_tracecov.py --I 5  --Nx 32 --spectrum gaussian --ell 1.0 --sigma_tot 0.3
          python burgers_spde_1d_tracecov.py --I 20 --Nx 64 --spectrum band --kc_ratio 0.3 --sigma_tot 0.2
        """
        u = irfft(uh, n=Nx)
        qh = rfft(u * u)
        qh *= kmask  # de-alias after product
        Nh = -0.5j * k * qh

        # Spectral noise: complex Gaussian with per-mode variance qk
        nf = len(k)
        zeta = rng.standard_normal(nf) + 1j * rng.standard_normal(nf)
        # endpoints real for rFFT representation
        zeta[0] = rng.standard_normal()
        if Nx % 2 == 0:
            zeta[-1] = rng.standard_normal()

        etah = np.sqrt(qk) * zeta  # per-step spectral kick (variance qk)
        # Euler–Maruyama with implicit L
        uh = (uh + dt * Nh + np.sqrt(dt) * etah) / denom
        return uh

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
