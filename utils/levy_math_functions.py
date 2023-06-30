import numpy as np
from numpy import cosh, sinh, exp, real, pi, abs
from scipy.special import gamma as gammafnc
from scipy.special import hankel1, hankel2, gammainc, gammaincc


def incgammau(s, x):
    return gammaincc(s, x) * gammafnc(s)


def incgammal(s, x):
    return gammainc(s, x) * gammafnc(s)


def psi(x, alpha, lambd):
    return -alpha * (cosh(x) - 1) - lambd * (exp(x) - x - 1)


def dpsi(x, alpha, lambd):
    return -alpha * sinh(x) - lambd * (exp(x) - 1)


def hankel_squared(lam, z):
    return real(hankel1(lam, z) * hankel2(lam, z))


def get_z0_H0(lambd):
    a = pi * (1.0 - 2.0 * abs(lambd)) ** 2
    b = gammafnc(np.abs(lambd)) ** 2
    c = 1 / (1 - 2 * np.abs(lambd))
    z1 = (a / b) ** c
    H0 = z1 * hankel_squared(np.abs(lambd), z1)
    return z1, H0


def g(x, sd, td, f1, f2):
    """ Helper function for GIG marginal sample generator """
    a = 0
    b = 0
    c = 0
    if (x >= -sd) and (x <= td):
        a = 1
    elif (x > td):
        b = f1
    elif (x < -sd):
        c = f2
    return a + b + c
