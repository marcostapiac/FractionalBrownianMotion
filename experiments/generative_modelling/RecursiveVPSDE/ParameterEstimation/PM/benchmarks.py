#!/usr/bin/env python
# coding: utf-8
import sys
import time
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from configs import project_config
import tracemalloc


def gaussian_kernel(bw, x):
    return np.exp(-0.5 * (x / bw) ** 2) / (bw * ((2 * np.pi) ** 0.5))  ##


def memory_check(str):
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print(f"{str}: [ Top 10 memory allocations ]")
    for stat in top_stats[:10]:
        print(stat)


end_diff_time = 1.
start_diff_time = 0.
ts_length = 256
Delta = 1. / ts_length
# Drift parameters
mean_rev = 1.

# In[4]:
# tracemalloc.start()
num_paths = 10000
ddata = np.load(project_config.ROOT_DIR + "data/fSin_samples_H05_T256_10Rev_10Diff_00Init.npy")
idxs = np.arange(ddata.shape[0])
path_observations = ddata[:num_paths, :]
prevPath_observations = np.concatenate([np.zeros((path_observations.shape[0], 1)), path_observations[:, :-1]], axis=1)
path_incs = np.concatenate([path_observations[:, [0]], np.diff(path_observations, axis=1)], axis=1)
assert (prevPath_observations.shape == path_observations.shape == path_incs.shape)
assert (prevPath_observations.shape[1] == ts_length)
del path_observations


def IID_NW_estimator(prevPath_observations, path_incs, bw, x, end_diff_time, start_diff_time):
    N, n = prevPath_observations.shape
    k = prevPath_observations[:, :, np.newaxis] - x
    kernel_weights_unnorm = gaussian_kernel(bw=bw, x=k)
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0)) / (N * n)
    k = kernel_weights_unnorm * path_incs[:, :, np.newaxis]
    numerator = np.sum(k, axis=(1, 0)) / N * (
            end_diff_time - start_diff_time)
    return numerator / denominator


def vect_IID_NW_estimator(prevPath_observations, path_incs, bw, x, end_diff_time, start_diff_time):
    N, N_minus1, n = prevPath_observations.shape
    t0 = time.time()
    k = prevPath_observations[:, :, :, np.newaxis] - x[:, np.newaxis, np.newaxis, :]
    for i in range(0):
        submat = prevPath_observations[i, :, :]
        x1 = x[i, :]
        k1 = submat[:, :, np.newaxis] - x1
        k2 = k[i, :, :, :]
        assert (np.allclose(k1, k2))
    print(time.time() - t0)
    t0 = time.time()
    kernel_weights_unnorm = gaussian_kernel(bw=bw, x=k)
    print(time.time() - t0)
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 2)) / (N * n)
    k = kernel_weights_unnorm * path_incs[:, :, :, np.newaxis]
    for i in range(0):
        k1 = k[i, :, :, :]
        k2 = kernel_weights_unnorm[i, :, :, :] * path_incs[i, :, :, np.newaxis]
        assert (np.allclose(k1, k2))
    numerator = np.sum(k, axis=(1, 2)) / N * (end_diff_time - start_diff_time)
    return numerator / denominator


def exclude_row(matrix):
    M, N = matrix.shape
    matrix = np.repeat(matrix[np.newaxis, :, :], M, axis=0)
    # Create a mask for rows to exclude
    mask = np.ones((M, M), dtype=bool)  # Start with all rows included
    np.fill_diagonal(mask, False)
    mask = np.concatenate([mask[:, :, np.newaxis]] * N, axis=-1)
    # Apply mask to create (M, M-1, N)
    result = matrix[mask].reshape(M, M - 1, N)
    return result


def compute_cv_for_bw_per_path(i, _bw, prevPath_observations, path_incs):
    N = prevPath_observations.shape[0]
    mask = np.arange(N) != i
    sys.stdout.flush()
    estimator = IID_NW_estimator(
        prevPath_observations=prevPath_observations[mask, :],
        path_incs=path_incs[mask, :],
        bw=_bw,
        x=prevPath_observations[i, :],
        end_diff_time=end_diff_time,
        start_diff_time=start_diff_time
    )
    residual = estimator ** 2 * Delta - 2 * estimator * path_incs[i, :]
    cv = np.sum(residual)
    if np.isnan(cv):
        return np.inf
    return cv


def compute_cv_for_bw(_bw, prevPath_observations, path_incs):
    N = prevPath_observations.shape[0]
    cvs = Parallel(n_jobs=30, backend="loky")(
        delayed(compute_cv_for_bw_per_path)(i, _bw, prevPath_observations, path_incs) for i in range(N))
    return np.sum(cvs)


assert (prevPath_observations.shape[1] * Delta == (end_diff_time - start_diff_time))


def compute_cv_for_bw_vectorised(_bw, prevPath_observations, path_incs, bw_obs, bw_incs):
    vec_estimator = vect_IID_NW_estimator(bw_obs, bw_incs, _bw, prevPath_observations, end_diff_time, start_diff_time)
    residual = vec_estimator ** 2 * Delta - 2 * vec_estimator * path_incs
    return {_bw: np.sum(residual)}


bws = np.logspace(-2, 0, 20)
if num_paths > 1000:
    bw_obs = exclude_row(prevPath_observations)
    bw_incs = exclude_row(path_incs)
    list_cvs = Parallel(n_jobs=1, backend="loky")(
        delayed(compute_cv_for_bw_vectorised)(_bw, prevPath_observations, path_incs, bw_obs, bw_incs) for _bw in bws)
    CVs = {}
    # Merge all dictionaries into the single dictionary
    for d in list_cvs:
        CVs.update(d)
    print(CVs)
else:
    mask = np.ones(prevPath_observations.shape[0], dtype=bool)
    CVs = np.zeros(len(bws))
    for h in tqdm(range(bws.shape[0])):
        CVs[h] = compute_cv_for_bw(bws[h], prevPath_observations, path_incs)

# In[ ]:
bw = bws[np.argmin(CVs.values())]
print(bw, CVs)

# In[45]:

num_Xs = 256
num_ests = 500
Xs = np.linspace(-2, 2, num_Xs)
drift_hats = np.zeros((num_Xs, num_ests))
for k in tqdm(range(num_ests)):
    path_observations = ddata[np.random.choice(idxs, size=num_paths, replace=False), :]
    prevPath_observations = np.concatenate([np.zeros((path_observations.shape[0], 1)), path_observations[:, :-1]],
                                           axis=1)
    path_incs = np.concatenate([path_observations[:, [0]], np.diff(path_observations, axis=1)], axis=1)
    del path_observations
    drift_hats[:, k] = IID_NW_estimator(prevPath_observations=prevPath_observations, bw=bw, x=Xs, path_incs=path_incs,
                                        end_diff_time=end_diff_time, start_diff_time=start_diff_time)

save_path = (
        project_config.ROOT_DIR + f"experiments/results/TS_benchmark_fSin_DriftEvalExp_{round(bw, 4)}bw_{num_paths}NPaths").replace(
    ".", "")
# In[50]:
np.save(save_path + "_driftHats.npy", drift_hats)
