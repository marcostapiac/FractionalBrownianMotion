from configs.project_config import NoneType
from typing import Union

import numpy as np
from numpy import abs
from scipy.special import gamma as gammafnc


class FractionalBrownianNoise:
    def __init__(self, H, rng=np.random.default_rng()):
        self.H = H
        self.rng = rng

    def scale_to_unit_time_interval(self, N: int, samples: np.ndarray):
        assert (samples.shape[0] == N)
        return np.power(1. / N, self.H) * samples

    def covariance(self, lag: int) -> float:
        """ Covariance and variance function for INCREMENTS ONLY with delta_T = 1.0 """
        return 0.5 * (abs(lag + 1) ** (2 * self.H) + abs(lag - 1) ** (2 * self.H) - 2 * abs(lag) ** (2 * self.H))

    def fBm_covariance(self, t1: int, t2: int) -> float:
        """ Covariance and variance function for MOTION with delta_T = 1.0 """
        return 0.5 * (abs(t1) ** (2. * self.H) + abs(t2) ** (2. * self.H) - abs(t1 - t2) ** (2. * self.H))

    def spectral_helper(self, x: np.ndarray) -> np.ndarray:
        """ Approximation of spectral density based on Paxon """
        B3 = np.squeeze(np.sum(
            [np.power((2. * np.pi * j + x), -2. * self.H - 1) + np.power((2. * np.pi * j - x), -2. * self.H - 1) for j
             in
             range(1, 4)]))
        B3 += np.squeeze(np.sum(
            [np.power((2. * np.pi * j + x), -2. * self.H) + np.power((2. * np.pi * j - x), -2. * self.H) for j in
             range(3, 5)])) / (
                      8. * self.H * np.pi)
        return (1.0002 - 0.000134 * x) * (B3 - np.power(2., -7.65 * self.H - 7.4))

    def spectral_density(self, x: np.ndarray) -> float:
        return 2. * np.sin(np.pi * self.H) * gammafnc(2. * self.H + 1.) * (1. - np.cos(x)) * (
                np.power(np.abs(x), -1. - 2. * self.H) + self.spectral_helper(x))

    def hosking_simulation(self, N_samples: int) -> np.ndarray:
        samples = np.atleast_2d([self.rng.normal()])  # Row vector
        d = np.atleast_2d([self.covariance(lag=1)]).T  # Column vector
        c = np.atleast_2d([d[0, 0]]).T  # Column vector
        mu = d[0, 0] * samples[0, 0]
        sigma2 = 1. - d[0, 0] ** 2
        for i in range(N_samples - 1):
            new_sample = np.atleast_2d([mu + np.sqrt(sigma2) * self.rng.normal()])
            samples = np.hstack([samples, new_sample])
            assert (samples.shape[0] == 1)
            assert (samples.shape[1] == i + 2)
            assert (c[-1, -1] == np.flip(c, axis=0)[0, 0])
            tau = np.squeeze(d.T @ np.flip(c, axis=0))  # Flip column vector vertically and ensure scalar (not vector)
            innovation = self.covariance(lag=i + 2) - tau
            phi = sigma2 ** (-1) * innovation
            sigma2 = sigma2 - sigma2 ** (-1) * (innovation ** 2)
            d = np.vstack([d - phi * np.flip(d, axis=0), phi])  # Flip 'd' vertically
            mu = np.squeeze(np.flip(samples, axis=1) @ d)  # Flip row vector horizontally
            c = np.vstack([c, self.covariance(lag=i + 2)])  # No need to tranpose if c is already column
        return self.scale_to_unit_time_interval(N_samples, np.squeeze(samples))

    def circulant_simulation(self, N_samples: int, scaleUnitInterval:bool=True, gaussRvs: Union[NoneType, np.ndarray] = None) -> np.ndarray:
        assert (type(N_samples) == int and (gaussRvs is not None and len(gaussRvs) == 2 * N_samples) or not gaussRvs)
        W = np.atleast_2d([complex(0., 0.)] * (2 * N_samples)).T
        assert (W.shape[1] > 0 and W.shape[0] == 2 * N_samples)
        if gaussRvs is None:
            W[0, 0], W[N_samples, 0] = self.rng.normal(), self.rng.normal()
            for j in range(1, N_samples):
                V1, V2 = self.rng.normal(), self.rng.normal()
                W[2 * N_samples - j, 0] = complex(V1, -V2) / np.sqrt(2)
                W[j, 0] = complex(V1, V2) / np.sqrt(2)
        else:
            W[0, 0], W[N_samples, 0] = gaussRvs[0], gaussRvs[N_samples]
            for j in range(1, N_samples):
                V1, V2 = gaussRvs[j], gaussRvs[2 * N_samples - j]
                W[2 * N_samples - j, 0] = complex(V1, -V2) / np.sqrt(2)
                W[j, 0] = complex(V1, V2) / np.sqrt(2)
        c = np.array([self.covariance(0 - j) for j in range(N_samples)])  # TODO: Check endpoint j == 2N-1
        c = np.append(c, 0.)
        c = np.append(c, np.squeeze(np.array([self.covariance(N_samples - 1 - j) for j in range(N_samples - 1)])))
        lambdas = np.fft.ifft(c)  # Should be real
        dotPs = np.diag(np.atleast_2d(np.sqrt(lambdas)).T @ W.T)
        Zs = np.fft.fft(dotPs)  # TOD0: Check implementation divides by root of length of dotPs
        if scaleUnitInterval: return self.scale_to_unit_time_interval(N_samples, np.real(Zs[:N_samples]))
        return np.real(Zs[:N_samples])

    def crmd_simulation(self, N_samples: int, l: int, r: int) -> np.ndarray:
        g = int(np.ceil(np.log2(N_samples)))
        Zprev = [self.rng.normal()]
        for i in range(1, g + 1):
            Zcurr = []
            expConst = 2 ** (i - 1)  # 2^i/2
            expHConst = 2 ** (-2 * self.H * i)
            left_indices = np.array([])
            for j in range(expConst):
                right_indices = np.unique([min(j + 1 + n, expConst) for n in range(r)])
                if j != 0: left_indices = np.unique([max(1, 2 * j + 1 - l + m) for m in range(l)])
                M = np.atleast_2d(
                    np.hstack([[Zprev[i - 1] for i in right_indices], [Zcurr[i - 1] for i in left_indices]])).T
                c = np.atleast_2d(np.hstack([[self.covariance(2. * n - (2 * j + 1)) + self.covariance(
                    2. * n - (2 * j + 1) - 1) for n in right_indices],
                                             [self.covariance(m - (2 * j + 1)) for m in left_indices]])).T
                if left_indices.size == 0:
                    upper = (2. ** (2 * self.H)) * np.array(
                        [[self.covariance(m2 - m1) for m1 in right_indices] for m2 in right_indices])
                elif right_indices.size == 0:
                    upper = np.array([[self.covariance(m2 - m1) for m1 in left_indices] for m2 in left_indices])
                else:
                    up1 = (2. ** (2 * self.H)) * np.array(
                        [[self.covariance(m2 - m1) for m1 in right_indices] for m2 in right_indices])
                    up2 = np.array(
                        [[self.covariance(2 * m1 - m2) + self.covariance(-m2 + 2 * (m1) - 1) for m1 in
                          right_indices]
                         for m2 in left_indices]).T
                    up3 = np.array(
                        [[self.covariance(m2 - 2 * m1) + self.covariance(m2 - 2 * (m1) + 1) for m1 in
                          left_indices]
                         for m2 in right_indices]).T
                    up4 = np.array([[self.covariance(m2 - m1) for m1 in left_indices] for m2 in left_indices])
                    upper = np.atleast_2d(np.vstack([np.hstack([up1, up2]), np.hstack([up3, up4])])).T
                inv = np.linalg.inv(upper)
                mu = c.T @ inv @ M
                sigma2 = expHConst * (1. - c.T @ inv @ c)
                U = self.rng.normal()
                a = np.sqrt(sigma2) * U
                Z1 = mu + a
                Z2 = Zprev[j] - Z1
                Zcurr.append(np.squeeze(Z1))
                Zcurr.append(np.squeeze(Z2))
            Zprev = Zcurr
        return self.scale_to_unit_time_interval(N_samples, np.array(
            Zprev[:N_samples]))  # Remember to multiply by 2**(self.H*g) if desire increments over deltaT = 1.

    def paxon_simulation(self, N_samples: int) -> np.ndarray:
        """ Spectral method for approximate samples of Fractional Brownian Noise for N = 2**k """
        fourier_coeffs = np.array([0.])
        for k in range(1, int(np.ceil(N_samples / 2))):  # Range must be integers
            r = self.rng.exponential(1.)
            phi = self.rng.uniform(0., 2 * np.pi)
            f = self.spectral_density(2. * np.pi * k / N_samples)
            fourier_coeffs = np.append(fourier_coeffs, np.sqrt(r * f / N_samples) * (np.cos(phi) + 1j * np.sin(phi)))
        fourier_coeffs = np.append(fourier_coeffs,
                                   np.sqrt(self.spectral_density(np.pi) / (2. * N_samples)) * self.rng.normal())
        cs = [np.conjugate(fourier_coeffs[N_samples - k]) for k in range(int(np.ceil(N_samples / 2)) + 1, N_samples)]
        fourier_coeffs = np.append(fourier_coeffs, cs)
        samples = np.fft.fft(fourier_coeffs)  # Samples should be real
        return self.scale_to_unit_time_interval(N_samples, np.real(samples))
