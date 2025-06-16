#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm

from configs import project_config
from src.classes.ClassFractionalBiPotential import FractionalBiPotential
from configs.RecursiveVPSDE.Markovian_fBiPotDDims.recursive_Markovian_PostMeanScore_fBiPot12Dims_T256_H05_tl_110data_StbleTgt import get_config
from utils.drift_evaluation_functions import IID_NW_multivar_estimator


def gaussian_kernel(bw, x):
    return norm.pdf(x / bw) / bw


def rmse_ignore_nans(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # Ignore NaNs in both arrays
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


# In[18]:


config = get_config()

num_paths = 10240
num_time_steps = config.ts_length
isUnitInterval = True
diff = config.diffusion
initial_state = config.initState
rvs = None
H = config.hurst
deltaT = config.deltaT
t0 = config.t0
t1 = deltaT * num_time_steps
try:
    is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :, :]
    is_path_observations = np.concatenate(
        [np.repeat(np.array(config.initState).reshape((1, 1, config.ndims)), is_path_observations.shape[0], axis=0),
         is_path_observations], axis=1)
    assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.ndims)
except (FileNotFoundError, AssertionError) as e:
    print(e)
    fLnz = FractionalBiPotential(num_dims=config.ndims, const=config.const, quartic_coeff=config.quartic_coeff,
                                 quad_coeff=config.quad_coeff,
                                 diff=diff, X0=initial_state)
    is_path_observations = np.array(
        [fLnz.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, X0=initial_state, Ms=None, gaussRvs=rvs,
                               t0=t0, t1=t1, isUnitInterval=True) for _ in (range(num_paths))]).reshape(
        (num_paths, config.ts_length + 1, config.ndims))
    np.save(config.data_path, is_path_observations[:, 1:, :])
    assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.ndims)

is_idxs = np.arange(is_path_observations.shape[0])
path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :]
# We note that we DO NOT evaluate the drift at time t_{0}=0
# We therefore remove the first element of path_observations since it includes X_{t_{0}} = X_{0}
# We also remove the last element since we never evaluate the drift at that point
t0 = deltaT
prevPath_observations = path_observations[:, 1:-1]
# We compute the path incs with respect to the prevPath_observations (since X_{t_{0}} != X_{0})
path_incs = np.diff(path_observations, axis=1)[:, 1:]
assert (prevPath_observations.shape == path_incs.shape)
assert (path_incs.shape[1] == config.ts_length - 1)
assert (path_observations.shape[1] == prevPath_observations.shape[1] + 2)



assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))

# Note that because b(x) = sin(x) is bounded, we take \epsilon = 0 hence we have following h_max
eps = 0.
log_h_min = np.log10(np.power(float(config.ts_length - 1), -(1. / (2. - eps))))
print(log_h_min)


def compute_cv_for_bw_per_path(i, inv_H):
    N = prevPath_observations.shape[0]
    mask = np.arange(N) != i  # Leave-one-out !
    estimator = IID_NW_multivar_estimator(
        prevPath_observations=prevPath_observations[mask, :],
        path_incs=path_incs[mask, :],
        inv_H=inv_H,
        x=prevPath_observations[i, :],
        t1=t1,
        t0=t0,
        truncate=False
    )
    residual = estimator ** 2 * deltaT - 2 * estimator * path_incs[i, :]
    cv = np.sum(residual)
    if np.isnan(cv):
        return np.inf
    return cv


def compute_cv_for_bw(_bw):
    N = prevPath_observations.shape[0]
    cvs = Parallel(n_jobs=14)(delayed(compute_cv_for_bw_per_path)(i, _bw) for i in (range(N)))
    # cvs = [compute_cv_for_bw_per_path(i, _bw) for i in range(N)]
    return np.sum(cvs)


grid_1d = np.logspace(-4, -0.05, 50)
bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
assert (bws.shape == (50, config.ndims))
# CVs = np.zeros(len(bws))
# for h in tqdm(range(bws.shape[0])):
#    CVs[h] = compute_cv_for_bw(bws[h])

# bw = bws[np.argmin(CVs)]
# print(CVs)

Xshape = 256
Xs = np.concatenate([np.linspace(-5, 5, num=Xshape).reshape(-1,1), np.linspace(-4.7, 4.7, num=Xshape).reshape(-1,1), \
                                     np.linspace(-4.4, 4.4, num=Xshape).reshape(-1,1), np.linspace(-4.2, 4.2, num=Xshape).reshape(-1,1), \
                                     np.linspace(-4.05, 4.05, num=Xshape).reshape(-1,1), np.linspace(-3.9, 3.9, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.7, 3.7, num=Xshape).reshape(-1,1), np.linspace(-3.6, 3.6, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.55, 3.55, num=Xshape).reshape(-1,1),
                                     np.linspace(-3.48, 3.48, num=Xshape).reshape(-1,1), \
                                     np.linspace(-3.4, 3.4, num=Xshape).reshape(-1,1), np.linspace(-3.4, 3.4, num=Xshape).reshape(-1,1)],
                                    axis=1)
num_dhats = 100
for bw in bws:
    inv_H = np.diag(np.power(bw, -2))
    norm_const = 1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H)))

    unif_is_drift_hats = np.zeros((Xshape, num_dhats, config.ts_dims))
    for k in tqdm(range(num_dhats)):
        is_ss_path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :]
        is_prevPath_observations = is_ss_path_observations[:, 1:-1]
        is_path_incs = np.diff(is_ss_path_observations, axis=1)[:, 1:]
        unif_is_drift_hats[:, k, :] = IID_NW_multivar_estimator(prevPath_observations=is_prevPath_observations, inv_H=inv_H, x=Xs,
                                                    path_incs=is_path_incs, t1=t1, t0=t0, truncate=True, norm_const=norm_const)

    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fBiPot_{config.ndims}DDims_DriftEvalExp_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c").replace(
        ".", "")
    np.save(save_path + "_isdriftHats.npy", unif_is_drift_hats)

# In[ ]:


# In[ ]:
