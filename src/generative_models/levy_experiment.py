from levy_processes.mean_mixture_processes import NormalGammaProcess
import numpy as np


def simulate_process(T:int)->np.array:
    mu_W = 1.0
    var_W = 2.0
    beta = .5
    C = 1.0
    process = NormalGammaProcess(beta=beta, C=C, mu=0., mu_W=mu_W, var_W=var_W)
    process.s
