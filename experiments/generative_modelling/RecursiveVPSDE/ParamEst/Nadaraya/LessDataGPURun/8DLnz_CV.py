#!/usr/bin/env python
# coding: utf-8

# In[1]:


from multiprocessing import shared_memory

import numpy as np
from tqdm import tqdm

from configs.RecursiveVPSDE.Markovian_8DLorenz.recursive_Markovian_PostMeanScore_8DLorenz_Stable_T256_H05_tl_110data_StbleTgt import \
    get_config

import math
import numpy as np
import torch


def _get_device(device_str: str | None = None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def IID_NW_multivar_estimator_gpu(
    prevPath_observations: torch.Tensor,  # (N,n,d) float32 CUDA
    path_incs: torch.Tensor,              # (N,n,d) float32 CUDA
    inv_H: torch.Tensor,                  # (d,) (diag) or (d,d) float32 CUDA
    norm_const: float,                    # same meaning as your CPU code
    x: torch.Tensor,                      # (M,d) float32 CUDA
    t1: float,
    t0: float,
    truncate: bool = True,
    M_tile: int = 32,                     # micro-batch states
    Nn_tile: int | None = 512_000,        # micro-batch samples (None => full)
    stable: bool = True,
) -> torch.Tensor:
    """
    Returns: (M,d) float32 CUDA tensor (keeps all heavy ops on LongerTimes_GPU).
    Matches your scaling:
      denom = sum(w)/(N*n)
      numer = (sum(w * incs)/N) * (t1 - t0)
    """
    #assert prevPath_observations.is_cuda and path_incs.is_cuda and x.is_cuda
    assert prevPath_observations.dtype == torch.float32
    assert path_incs.dtype == torch.float32
    assert x.dtype == torch.float32

    N, n, d = prevPath_observations.shape
    Nn = N * n
    if Nn_tile is None or Nn_tile > Nn:
        Nn_tile = Nn

    # Flatten once
    mu = prevPath_observations.reshape(Nn, d).contiguous()  # (Nn,d)
    dX = path_incs.reshape(Nn, d).contiguous()              # (Nn,d)

    # Diagonal vs full inv_H
    diag = (inv_H.ndim == 1)
    if diag:
        A = inv_H                                           # (d,)
        muAh = mu * A                                       # (Nn,d)
        mu_quad = (mu * muAh).sum(-1)                       # (Nn,)
        def xAh(X): return X * A
    else:
        A = inv_H                                           # (d,d)
        muAh = mu @ A                                       # (Nn,d)
        mu_quad = (mu * muAh).sum(-1)                       # (Nn,)
        # Sanity: PD
        sign, _ = torch.linalg.slogdet(A)
        if sign.item() <= 0:
            raise ValueError("inv_H must be positive definite.")

    # Use log(norm_const) directly to match your CPU estimator
    log_norm_const = float(np.log(norm_const))

    M = x.size(0)
    # Accumulate in float64 for stability
    denom = torch.zeros(M,     dtype=torch.float64, device=x.device)
    numer = torch.zeros(M, d,  dtype=torch.float64, device=x.device)

    for m0 in range(0, M, M_tile):
        X = x[m0:m0 + M_tile]                     # (mb,d)
        XAh = xAh(X)                               # (mb,d)
        X_quad = (X * XAh).sum(-1)                 # (mb,)

        denom_tile = torch.zeros(X.size(0),    dtype=torch.float64, device=x.device)
        numer_tile = torch.zeros(X.size(0), d, dtype=torch.float64, device=x.device)

        if stable:
            lse_max = torch.full((X.size(0),), -torch.inf, dtype=torch.float32, device=x.device)
            # First pass: find max exponent per state (over all Nn tiles)
            for i0 in range(0, Nn, Nn_tile):
                muq_i  = mu_quad[i0:i0 + Nn_tile]             # (bn,)
                muAh_i = muAh[i0:i0 + Nn_tile]                # (bn,d)
                cross  = muAh_i @ X.t()                       # (bn,mb)
                expo   = log_norm_const - 0.5 * (muq_i[:, None] + X_quad[None, :] - 2.0 * cross)
                lse_max = torch.maximum(lse_max, expo.max(dim=0).values)

        # Second pass: accumulate with stabilization (or plain)
        for i0 in range(0, Nn, Nn_tile):
            muAh_i = muAh[i0:i0 + Nn_tile]                    # (bn,d)
            muq_i  = mu_quad[i0:i0 + Nn_tile]                 # (bn,)
            dX_i   = dX[i0:i0 + Nn_tile]                      # (bn,d)

            cross = muAh_i @ X.t()                            # (bn,mb)
            expo  = log_norm_const - 0.5 * (muq_i[:, None] + X_quad[None, :] - 2.0 * cross)

            if stable:
                w = torch.exp(expo - lse_max[None, :])        # (bn,mb)
            else:
                w = torch.exp(expo)

            denom_tile += w.sum(dim=0, dtype=torch.float64) / (N * n)
            numer_tile += (w.t() @ dX_i).to(torch.float64) * ((t1 - t0) / N)

        if stable:
            scale = torch.exp(lse_max.to(torch.float64))
            denom_tile *= scale
            numer_tile *= scale[:, None]

        denom[m0:m0 + X.size(0)] += denom_tile
        numer[m0:m0 + X.size(0)] += numer_tile

    est = torch.full((M, d), float('nan'), dtype=torch.float32, device=x.device)
    mask = (denom > 0) & torch.isfinite(denom) & torch.isfinite(numer).all(dim=1)
    est[mask] = (numer[mask] / denom[mask, None]).to(torch.float32)

    return est

config = get_config()
num_paths = 1024 if config.feat_thresh == 1. else 10240
assert num_paths == 1024
t0 = config.t0
deltaT = config.deltaT
t1 = deltaT * config.ts_length
# Drift parameters
diff = config.diffusion
initial_state = np.array(config.initState)
rvs = None
H = config.hurst
is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :, :]
is_path_observations = np.concatenate(
    [np.repeat(np.array(config.initState).reshape((1, 1, config.ndims)), is_path_observations.shape[0], axis=0),
     is_path_observations], axis=1)
assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.ndims)

is_idxs = np.arange(is_path_observations.shape[0])
path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :]
t0 = deltaT
prevPath_observations = path_observations[:, 1:-1, :]
path_incs = np.diff(path_observations, axis=1)[:, 1:, :]
assert (prevPath_observations.shape == path_incs.shape)
assert (path_incs.shape[1] == config.ts_length - 1)
assert (path_observations.shape[1] == prevPath_observations.shape[1] + 2)
assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))

# Note that because b(x) = sin(x) is bounded, we take \epsilon = 0 hence we have following h_max
eps = 0.
log_h_min = np.log10(np.power(float(config.ts_length - 1), -(1. / (2. - eps))))
print(log_h_min)


def compute_cv_for_bw_per_path(i, _bw, device, deltaT, path_incs, prevPath_observations):
    N = prevPath_observations.shape[0]
    mask = np.arange(N) != i  # Leave-one-out !
    maskedprevPath_observations=torch.tensor(prevPath_observations[mask, :], device=device, dtype=torch.float32)
    maskedpath_incs=torch.tensor(path_incs[mask, :], device=device, dtype=torch.float32)
    x = torch.tensor(prevPath_observations[i, :], device=device, dtype=torch.float32)
    Nn_tile = 512000
    M_tile = 2048
    inv_H = np.diag(np.power(_bw, -2))
    norm_const = float(1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H))))
    inv_H_np = np.asarray(inv_H)
    if inv_H_np.ndim == 2 and np.allclose(inv_H_np, np.diag(np.diag(inv_H_np))):
        inv_H_vec = np.diag(inv_H_np).astype(np.float32)
        inv_H = torch.as_tensor(inv_H_vec, device=device)
    else:
        inv_H = torch.as_tensor(inv_H_np.astype(np.float32), device=device)
    estimator = IID_NW_multivar_estimator_gpu(
        prevPath_observations=maskedprevPath_observations,
        path_incs=maskedpath_incs,
        inv_H=inv_H,
        norm_const=norm_const,
        x=x,
        t1=t1,
        t0=t0,
        truncate=True, Nn_tile=Nn_tile, M_tile=M_tile, stable=True
    )
    residual = estimator ** 2 * deltaT - 2 * estimator * torch.tensor(path_incs[i, :], device=device, dtype=torch.float32)
    cv = torch.sum(residual).detach().cpu().numpy()
    if np.isnan(cv):
        return np.inf
    return cv


grid_1d = np.logspace(-3.55, 4, 100)
bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
assert (bws.shape == (100, config.ndims))
CVs = np.zeros(len(bws))
device_id = _get_device()
for h in tqdm(range(bws.shape[0])):
    N = prevPath_observations.shape[0]
    cvs = []
    for i in range(N):
        cvs.append(compute_cv_for_bw_per_path(i, bws[h], device=device_id, path_incs=path_incs, prevPath_observations=prevPath_observations, deltaT=config.deltaT))
    CVs[h] = np.sum(cvs)
np.save()