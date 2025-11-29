#!/usr/bin/env python
# coding: utf-8
import math
import pandas as pd
import numpy as np
import torch

from configs import project_config
from configs.RecursiveVPSDE.Markovian_fSinLog.recursive_Markovian_PostMeanScore_fSinLog_LowFTh_T256_H05_tl_110data_StbleTgt_HIGHNOISE import \
    get_config


def basis_number_selection(paths, num_paths, num_time_steps, deltaT, t1, device_id):
    poss_Rs = torch.arange(1, 11)
    kappa = 1.  # See just above Section 5
    cvs = []
    for r in poss_Rs:
        basis = hermite_basis_GPU(R=r, paths=paths, device_id=device_id)
        try:
            Phi = construct_Phi_matrix(R=r, deltaT=deltaT, T=t1, basis=basis, paths=paths, device_id=device_id)
        except AssertionError:
            cvs.append(torch.inf)
            continue
        coeffs = estimate_coefficients(R=r, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=Phi,
                                       device_id=device_id)
        bhat = torch.pow(construct_Hermite_drift(basis=basis, coefficients=coeffs), 2)
        bhat_norm = torch.nanmean(torch.sum(bhat * deltaT / t1, dim=-1))
        inv_Phi = torch.linalg.inv(Phi)
        s = torch.sqrt(torch.max(torch.linalg.eigvalsh(inv_Phi @ inv_Phi.T)))
        if torch.pow(s, 0.25) * r > num_paths * t1:
            cvs.append(torch.inf)
        else:
            # Note that since we force \sigma = 1., then the m,sigma^2 matrix is all ones
            PPt = inv_Phi @ torch.ones_like(inv_Phi)
            s_p = torch.sqrt(torch.max(torch.linalg.eigvalsh(PPt @ PPt.T)))
            pen = kappa * s_p / (num_paths * num_time_steps * deltaT)
            cvs.append(-bhat_norm + pen)
    return poss_Rs[np.argmin(cvs)]


def hermite_basis_GPU(R, device_id, paths):
    paths = torch.as_tensor(paths, device=device_id, dtype=torch.float32)
    print(paths.shape)
    assert paths.ndim == 2 and paths.shape[0] >= 1
    N, D = paths.shape
    basis = torch.empty((N, D, R), device=device_id, dtype=torch.float32)
    polynomials = torch.empty_like(basis)
    for i in range(R):
        if i == 0:
            polynomials[:, :, i] = 1.0
        elif i == 1:
            polynomials[:, :, i] = paths
        else:
            polynomials[:, :, i] = 2.0 * paths * polynomials[:, :, i - 1] - 2.0 * (i - 1) * polynomials[:, :, i - 2]
        norm = (2.0 ** i * math.sqrt(math.pi) * math.factorial(i)) ** -0.5  # Python float is fine
        basis[:, :, i] = norm * polynomials[:, :, i] * torch.exp(-0.5 * paths ** 2)
    return basis


def construct_Z_vector(R, T, basis, paths):
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N = basis.shape[0]
    dXs = torch.diff(paths, dim=1) / T
    Z = torch.diagonal(basis.permute((2, 0, 1)) @ (dXs.T), dim1=1, dim2=2)
    assert (Z.shape == (R, N))
    Z = Z.mean(axis=-1, keepdims=True)
    assert (Z.shape == (R, 1)), f"Z vector is shape {Z.shape} but should be {(R, 1)}"
    return Z


def construct_Phi_matrix(R, deltaT, T, basis, device_id, paths):
    paths = torch.as_tensor(paths, device=device_id, dtype=torch.float32)
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N, _ = basis.shape[:2]
    deltaT /= T
    intermediate = deltaT * basis.permute((0, 2, 1)) @ basis
    assert intermediate.shape == (
        N, R, R), f"Intermediate matrix is shape {intermediate.shape} but should be {(N, R, R)}"
    #for i in range(N):
        #es = torch.linalg.eigvalsh(intermediate[i, :, :]) >= 0.
        # assert (torch.all(es)), f"Submat at {i} is not PD, for R={R}"
    Phi = deltaT * (basis.permute((0, 2, 1)) @ basis)
    assert (Phi.shape == (N, R, R))
    Phi = Phi.mean(axis=0, keepdims=False)
    assert (Phi.shape == (R, R)), f"Phi matrix is shape {Phi.shape} but should be {(R, R)}"
    # assert torch.all(torch.linalg.eigvalsh(Phi) >= 0.), f"Phi matrix is not PD"
    return Phi


def estimate_coefficients(R, deltaT, t1, basis, paths, device_id, Phi=None):
    paths = torch.as_tensor(paths, device=device_id, dtype=torch.float32)
    Z = construct_Z_vector(R=R, T=t1, basis=basis, paths=paths)
    if Phi is None:
        Phi = construct_Phi_matrix(R=R, deltaT=deltaT, T=t1, basis=basis, paths=paths, device_id=device_id)
    theta_hat = torch.linalg.solve(Phi, Z)
    assert (theta_hat.shape == (R, 1))
    return theta_hat


def construct_Hermite_drift(basis, coefficients):
    b_hat = (basis @ coefficients).squeeze(-1)
    assert (b_hat.shape == basis.shape[:2]), f"b_hat should be shape {basis.shape[:2]}, but has shape {b_hat.shape}"
    return b_hat


def _get_device(device_str: str | None = None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = get_config()
config.feat_thresh = 1./500
num_paths = 1024 if config.feat_thresh == 1. else 10240
assert num_paths == 10240
assert config.diffusion == 10.
num_time_steps = config.ts_length
isUnitInterval = True
diff = config.diffusion
initial_state = config.initState
rvs = None
H = config.hurst
deltaT = config.deltaT
t0 = config.t0
t1 = deltaT * num_time_steps
device_id = _get_device()
paths = torch.tensor(np.load(config.data_path, allow_pickle=True)[:num_paths, :], device=device_id, dtype=torch.float32)
paths = torch.concatenate(
    [torch.tensor(np.repeat((np.array(config.initState)).reshape((1, 1)), paths.shape[0], axis=0), device=device_id,
                  dtype=torch.float32),
     paths], dim=1)
assert paths.shape == (num_paths, config.ts_length + 1)

numXs = 1024  # config.ts_length
Xs = torch.linspace(-12., 12., numXs).reshape(1, -1)
def true_drifts(device_id, config, state):
    state = torch.tensor(state, device=device_id, dtype=torch.float32)
    drift = (-torch.sin(config.sin_space_scale * state) * torch.log(
        1 + config.log_space_scale * torch.abs(state)) / config.sin_space_scale)
    return drift[:, np.newaxis, :]

true_drift = true_drifts(device_id=device_id, config=config, state=Xs).cpu().squeeze().numpy().reshape((numXs, config.ndims))
# In[9]:
mses = {}
for R in np.arange(2, 41, 1):
    basis = hermite_basis_GPU(R=R, paths=paths, device_id=device_id)
    coeffs = (estimate_coefficients(R=R, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=None, device_id=device_id))
    basis = hermite_basis_GPU(R=R, paths=Xs, device_id=device_id)
    bhat = construct_Hermite_drift(basis=basis, coefficients=coeffs).cpu().squeeze().numpy().reshape((numXs, config.ndims))
    print(bhat.shape, true_drift.shape)
    mse = np.nanmean(np.sum(np.power(bhat - true_drift, 2), axis=-1), axis=-1)
    print(R, mse)
    mses[R] = [mse]
save_path = (
        project_config.ROOT_DIR + f"experiments/results/Hermite_fSinLog_DriftEvalExp_{num_paths}NPaths_{config.deltaT:.3e}dT_Diff{config.diffusion:.1f}").replace(
    ".", "")
mses = (pd.DataFrame(mses)).T
mses.columns = mses.columns.astype(str)
mses.to_parquet(save_path + "_MSEs.parquet", engine="fastparquet")
Ridx = np.argmin(mses.values.flatten())
R = np.arange(2, 41, 1)[Ridx]
basis = hermite_basis_GPU(R=R, paths=paths, device_id=device_id)
coeffs = (estimate_coefficients(R=R, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=None, device_id=device_id))
basis = hermite_basis_GPU(R=R, paths=Xs, device_id=device_id)
bhat = construct_Hermite_drift(basis=basis, coefficients=coeffs).cpu().squeeze().numpy()
np.save(save_path + "_drift_est.npy",bhat)
np.save(save_path + "_true_drift.npy",true_drift)

