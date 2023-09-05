import cProfile

import numpy as np

from src.classes.ClassFractionalCEV import FractionalCEV
from utils.distributions.CEV_multivar_posteriors import generate_V_matrix, fBn_covariance_matrix


def covariance_computation(muU=1., muX=1., gamma=1., X0=1., U0=0., H=0.8, N=2 ** 12, T=1e-3 * 2 ** 12):
    deltaT = T / N
    sigmaX = np.sqrt(muX * gamma / 0.55)
    m = FractionalCEV(muU=muU, gamma=gamma, muX=muX, sigmaX=sigmaX, X0=X0, U0=U0)
    Xs, Us = m.euler_simulation(H=H, N=N, deltaT=deltaT)  # Simulated data
    print("Starting computation")
    cProfile.runctx('g(vols=Xs, sigmaX=1., deltaT=deltaT, H=H, N=N)',
                    locals={'g': generate_V_matrix, 'Xs': Xs, 'deltaT': deltaT, 'H': H, 'N': N}, globals={},
                    sort='cumulative')


def SH_computation(N=2 ** 10, H=0.8):
    print("Starting computation")
    cProfile.runctx('g(H=H, N=N)', locals={'g': fBn_covariance_matrix, 'H': H, 'N': N}, globals={}, sort='cumulative')


def numpySum(N=2 ** 12, H=0.8):
    SH = fBn_covariance_matrix(H=H, N=N)
    cProfile.runctx('g(SH)', locals={'g': np.sum, 'SH': SH}, globals={}, sort='cumulative')
    cProfile.runctx('g.T@SH@g', locals={'g': np.ones(shape=SH.shape[0]), 'SH': SH}, globals={}, sort='cumulative')
    print(np.sum(SH), np.ones(shape=SH.shape[0]).T @ SH @ np.ones(shape=SH.shape[0]))


numpySum(N=2 ** 12, H=0.8)
