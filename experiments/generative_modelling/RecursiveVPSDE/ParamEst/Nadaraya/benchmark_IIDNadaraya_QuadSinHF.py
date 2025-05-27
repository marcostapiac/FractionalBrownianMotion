#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm

from configs import project_config
from src.classes.ClassFractionalQuadSin import FractionalQuadSin
from configs.RecursiveVPSDE.LSTM_fQuadSinHF.recursive_LSTM_PostMeanScore_fQuadSinHF_T256_H05_tl_110data import \
    get_config


def gaussian_kernel(bw, x):
    return norm.pdf(x / bw) / bw


def rmse_ignore_nans(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # Ignore NaNs in both arrays
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


# In[18]:


config = get_config()
print(config.deltaT)
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
print(deltaT, t0, t1)
try:
    is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :]
    is_path_observations = np.concatenate(
        [np.repeat(np.array(config.initState).reshape((1, 1)), is_path_observations.shape[0], axis=0),
         is_path_observations], axis=1)
    assert is_path_observations.shape == (num_paths, config.ts_length + 1)
except (FileNotFoundError, AssertionError) as e:
    print(e)
    fQuadSin = FractionalQuadSin(quad_coeff=config.quad_coeff, sin_coeff=config.sin_coeff,
                                 sin_space_scale=config.sin_space_scale, diff=diff, X0=initial_state)
    is_path_observations = np.array(
        [fQuadSin.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, isUnitInterval=isUnitInterval, X0=initial_state,
                                   Ms=None, gaussRvs=rvs,
                                   t0=t0, t1=t1) for _ in (range(num_paths * 10))]).reshape(
        (num_paths, config.ts_length + 1))
    np.save(config.data_path, is_path_observations[:, 1:])
    assert is_path_observations.shape == (num_paths, config.ts_length + 1)

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


def IID_NW_estimator(prevPath_observations, path_incs, bw, x, t1, t0, truncate):
    N, n = prevPath_observations.shape
    kernel_weights_unnorm = gaussian_kernel(bw=bw, x=prevPath_observations[:, :, np.newaxis] - x)
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0)) / (N * n)
    m = np.min(denominator)
    numerator = np.sum(kernel_weights_unnorm * path_incs[:, :, np.newaxis], axis=(1, 0)) / N * (t1 - t0)
    # This is the "truncated" discrete drift estimator to ensure appropriate risk bounds
    if truncate:
        estimator = numerator[denominator > m / 2.] / denominator[denominator > m / 2.]
    else:
        estimator = numerator / denominator
    return estimator


assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))

# Note that because b(x) = sin(x) is bounded, we take \epsilon = 0 hence we have following h_max
eps = 0.
log_h_min = np.log10(np.power(float(config.ts_length - 1), -(1. / (2. - eps))))
print(log_h_min)


def compute_cv_for_bw_per_path(i, _bw):
    N = prevPath_observations.shape[0]
    mask = np.arange(N) != i  # Leave-one-out !
    estimator = IID_NW_estimator(
        prevPath_observations=prevPath_observations[mask, :],
        path_incs=path_incs[mask, :],
        bw=_bw,
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


bws = np.logspace(-4, -0.05, 20)  # np.logspace(-2, -0.05, 20)
# CVs = np.zeros(len(bws))
# for h in tqdm(range(bws.shape[0])):
#    CVs[h] = compute_cv_for_bw(bws[h])

# bw = bws[np.argmin(CVs)]
# print(CVs)

numXs = 256  # config.ts_length
minx = -1.5
maxx = -minx
Xs = np.linspace(minx, maxx, numXs)
num_dhats = 10

for bw in bws:
    unif_is_drift_hats = np.zeros((numXs, num_dhats))

    for k in tqdm(range(num_dhats)):
        is_ss_path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :]
        # Remember t0 = deltaT so X_t1 = is_ss_path_observations[:, 2] not is_ss_path_observations[:, 1]
        is_prevPath_observations = is_ss_path_observations[:, 1:-1]
        is_path_incs = np.diff(is_ss_path_observations, axis=1)[:, 1:]
        unif_is_drift_hats[:, k] = IID_NW_estimator(prevPath_observations=is_prevPath_observations, bw=bw, x=Xs,
                                                    path_incs=is_path_incs, t1=t1, t0=t0, truncate=True)
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fQuadSinHF_DriftEvalExp_{round(bw, 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.ts_length}NumDPS").replace(
        ".", "")
    print(f"Save Path {save_path}\n")
    np.save(save_path + "_isdriftHats.npy", unif_is_drift_hats)
