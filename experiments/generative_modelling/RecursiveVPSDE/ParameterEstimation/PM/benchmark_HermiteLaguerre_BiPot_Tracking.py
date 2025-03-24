#!/usr/bin/env python
# coding: utf-8
from configs import project_config
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import eval_laguerre
from src.classes.ClassFractionalBiPotential import FractionalBiPotential
from configs.RecursiveVPSDE.LSTM_fBiPot.recursive_LSTM_PostMeanScore_fBiPot_T256_H05_tl_110data import get_config
from tqdm import tqdm

# # From Nonparametric drift estimation for IID paths of SDE (Comte et al 2020)

# In[2]:


config = get_config()

# In[3]:


num_paths = 10952
num_time_steps = config.ts_length
isUnitInterval = True
diff = config.diffusion
initial_state = 0.
rvs = None
H = config.hurst
deltaT = config.deltaT
t0 = config.t0
t1 = deltaT * num_time_steps
fBiPot = FractionalBiPotential(const=config.const, quartic_coeff=config.quartic_coeff, quad_coeff=config.quad_coeff,
                               diff=diff, X0=initial_state)
paths = np.array(
    [fBiPot.euler_simulation(H=H, N=num_time_steps, deltaT=deltaT, isUnitInterval=isUnitInterval, X0=initial_state,
                             Ms=None, gaussRvs=rvs,
                             t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
    (num_paths, num_time_steps + 1))


def hermite_basis(R, paths):
    assert (paths.shape[0] >= 1 and len(paths.shape) == 2)
    basis = np.zeros((paths.shape[0], paths.shape[1], R))
    polynomials = np.zeros((paths.shape[0], paths.shape[1], R))
    for i in range(R):
        if i == 0:
            polynomials[:, :, i] = np.ones_like(paths)
        elif i == 1:
            polynomials[:, :, i] = paths
        else:
            polynomials[:, :, i] = 2. * paths * polynomials[:, :, i - 1] - 2. * (i - 1) * polynomials[:, :, i - 2]
        basis[:, :, i] = np.power((np.power(2, i) * np.sqrt(np.pi) * math.factorial(i)), -0.5) * polynomials[:, :,
                                                                                                 i] * np.exp(
            -np.power(paths, 2) / 2.)
    return basis


def laguerre_basis(R, paths):
    basis = np.zeros((paths.shape[0], paths.shape[1], R))
    for i in range(R):
        basis[:, :, i] = np.sqrt(2.) * eval_laguerre(i, 2. * paths) * np.exp(-paths) * (paths >= 0.)
    return basis


def rmse_ignore_nans(y_true, y_pred):
    assert (y_true.shape == y_pred.shape and len(y_true.shape) == 1)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # Ignore NaNs in both arrays
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


def construct_Z_vector(R, T, basis, paths):
    print(basis.shape, paths.shape)
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N = basis.shape[0]
    dXs = np.diff(paths, axis=1) / T
    Z = np.diagonal(basis.transpose((2, 0, 1)) @ (dXs.T), axis1=1, axis2=2)
    assert (Z.shape == (R, N))
    Z = Z.mean(axis=-1, keepdims=True)
    assert (Z.shape == (R, 1)), f"Z vector is shape {Z.shape} but should be {(R, 1)}"
    return Z


def construct_Phi_matrix(R, deltaT, T, basis, paths):
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N, _ = basis.shape[:2]
    deltaT /= T
    intermediate = deltaT * basis.transpose((0, 2, 1)) @ basis
    assert intermediate.shape == (
        N, R, R), f"Intermidate matrix is shape {intermediate.shape} but shoould be {(N, R, R)}"
    for i in range(N):
        es = np.linalg.eigvalsh(intermediate[i, :, :]) >= 0.
        assert (np.all(es)), f"Submat at {i} is not PD, for R={R}"
    Phi = deltaT * (basis.transpose((0, 2, 1)) @ basis)
    assert (Phi.shape == (N, R, R))
    Phi = Phi.mean(axis=0, keepdims=False)
    assert (Phi.shape == (R, R)), f"Phi matrix is shape {Phi.shape} but should be {(R, R)}"
    assert np.all(np.linalg.eigvalsh(Phi) >= 0.), f"Phi matrix is not PD"
    return Phi


def estimate_coefficients(R, deltaT, t1, basis, paths, Phi=None):
    Z = construct_Z_vector(R=R, T=t1, basis=basis, paths=paths)
    if Phi is None:
        Phi = construct_Phi_matrix(R=R, deltaT=deltaT, T=t1, basis=basis, paths=paths)
    theta_hat = np.linalg.solve(Phi, Z)
    assert (theta_hat.shape == (R, 1))
    return theta_hat


def construct_drift(basis, coefficients):
    b_hat = (basis @ coefficients).squeeze(-1)
    assert (b_hat.shape == basis.shape[:2]), f"b_hat should be shape {basis.shape[:2]}, but has shape {b_hat.shape}"
    return b_hat


def basis_number_selection(paths, num_paths, num_time_steps, deltaT, t1):
    poss_Rs = np.arange(1, 11)
    kappa = 1.  # See just above Section 5
    cvs = []
    for r in poss_Rs:
        print(cvs, r)
        basis = hermite_basis(R=r, paths=paths)
        try:
            Phi = construct_Phi_matrix(R=r, deltaT=deltaT, T=t1, basis=basis, paths=paths)
        except AssertionError:
            cvs.append(np.inf)
            continue
        coeffs = estimate_coefficients(R=r, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=Phi)
        bhat = np.power(construct_drift(basis=basis, coefficients=coeffs), 2)
        bhat_norm = np.mean(np.sum(bhat * deltaT / t1, axis=-1))
        inv_Phi = np.linalg.inv(Phi)
        s = np.sqrt(np.max(np.linalg.eigvalsh(inv_Phi @ inv_Phi.T)))
        if np.power(s, 0.25) * r > num_paths * t1:
            cvs.append(np.inf)
        else:
            # Note that since we force \sigma = 1., then the m,sigma^2 matrix is all ones
            PPt = inv_Phi @ np.ones_like(inv_Phi)
            s_p = np.sqrt(np.max(np.linalg.eigvalsh(PPt @ PPt.T)))
            pen = kappa * s_p / (num_paths * num_time_steps * deltaT)
            cvs.append(-bhat_norm + pen)
    return poss_Rs[np.argmin(cvs)]


# R = basis_number_selection(paths=paths, num_paths=num_paths, num_time_steps=num_time_steps, deltaT=deltaT, t1=t1)
# print(R)
numXs = 256  # config.ts_length

def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -(4. * config.quartic_coeff * np.power(prev, 3) + 2. * config.quad_coeff * prev + config.const)
    return drift_X[:, np.newaxis, :]


for R in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
    basis = hermite_basis(R=R, paths=paths)
    coeffs = (estimate_coefficients(R=R, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=None))
    num_time_steps = 100
    num_state_paths = 10
    true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    # Initialise the "true paths"
    true_states[:, [0], :] = config.initState
    # Initialise the "global score-based drift paths"
    global_states[:, [0], :] = config.initState
    # Initialise the "local score-based drift paths"
    local_states[:, [0], :] = config.initState

    for i in tqdm(range(1, num_time_steps + 1)):
        eps = np.random.randn(num_state_paths, 1, config.ndims) * np.sqrt(deltaT)
        assert (eps.shape == (num_state_paths, 1, config.ndims))
        true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)
        global_basis = hermite_basis(R=R, paths=global_states[:, i - 1, :])
        global_mean = construct_drift(basis=global_basis, coefficients=coeffs)[:, np.newaxis, :]
        local_basis = hermite_basis(R=R, paths=true_states[:, i - 1, :])
        local_mean = construct_drift(basis=local_basis, coefficients=coeffs)[:, np.newaxis, :]
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/Hermite_fBiPot_DriftTracking_{R}R_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c").replace(
        ".", "")
    np.save(save_path + "_true_states.npy", true_states)
    np.save(save_path + "_global_states.npy", global_states)
    np.save(save_path + "_local_states.npy", local_states)
