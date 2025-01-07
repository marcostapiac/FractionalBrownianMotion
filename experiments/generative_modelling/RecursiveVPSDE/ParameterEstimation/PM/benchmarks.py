#!/usr/bin/env python
# coding: utf-8
import sys
import time
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from tqdm import tqdm

from configs import project_config


def gaussian_kernel(bw, x):
    return norm.pdf(x / bw) / bw


end_diff_time = 1.
start_diff_time = 0.
ts_length = 256
Delta = 1. / ts_length
# Drift parameters
mean_rev = 1.

# In[4]:

num_paths = 10000
ddata = np.load(project_config.ROOT_DIR + "data/fSin_samples_H05_T256_10Rev_10Diff_00Init.npy")
idxs = np.arange(ddata.shape[0])
path_observations = ddata[:num_paths,:]
prevPath_observations = np.concatenate([np.zeros((path_observations.shape[0], 1)), path_observations[:, :-1]], axis=1)
path_incs = np.concatenate([path_observations[:, [0]], np.diff(path_observations, axis=1)], axis=1)
assert (prevPath_observations.shape == path_observations.shape == path_incs.shape)
assert (prevPath_observations.shape[1] == ts_length)
del path_observations


# In[5]:


def IID_NW_estimator(prevPath_observations, path_incs, bw, x, end_diff_time, start_diff_time):
    N, n = prevPath_observations.shape
    kernel_weights_unnorm = gaussian_kernel(bw=bw, x=prevPath_observations[:, :, np.newaxis] - x)
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0)) / (N * n)
    numerator = np.sum(kernel_weights_unnorm * path_incs[:, :, np.newaxis], axis=(1, 0)) / N * (
            end_diff_time - start_diff_time)
    return numerator / denominator


assert (prevPath_observations.shape[1] * Delta == (end_diff_time - start_diff_time))


# In[7]:

def compute_cv_for_bw_per_path(i, _bw, prevPath_observations, path_incs):
    t0 = time.time()
    N = prevPath_observations.shape[0]
    mask = np.arange(N) != i
    print(i)
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
    print(time.time()-t0)
    return cv


def compute_cv_for_bw(_bw, prevPath_observations, path_incs):
    N = prevPath_observations.shape[0]
    print(f"Starting: {_bw}\n")
    t0 = time.time()
    cvs = Parallel(n_jobs=15)(delayed(compute_cv_for_bw_per_path)(i, _bw, prevPath_observations, path_incs) for i in range(N))
    print(time.time()-t0)
    return np.sum(cvs)


bws = np.logspace(-2, 0, 20)
mask = np.ones(prevPath_observations.shape[0], dtype=bool)
CVs = np.zeros(len(bws))
for h in tqdm(range(bws.shape[0])):
    CVs[h] = compute_cv_for_bw(bws[h], prevPath_observations, path_incs)

# In[ ]:


bw = bws[np.argmin(CVs)]
print(bw, CVs)

# In[45]:

num_Xs = 256
num_ests = 50000
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
        project_config.ROOT_DIR + f"experiments/results/TS_benchmark_fSin_DriftEvalExp_{round(bw, 4)}bw").replace(
    ".", "")
# In[50]:
np.save(save_path + "_driftHats.npy", drift_hats)
