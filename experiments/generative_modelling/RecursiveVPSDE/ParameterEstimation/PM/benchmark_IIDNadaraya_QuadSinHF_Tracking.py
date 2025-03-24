#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from configs import project_config
from configs.RecursiveVPSDE.LSTM_fQuadSinHF.recursive_LSTM_PostMeanScore_fQuadSinHF_T256_H05_tl_110data import get_config
from src.classes.ClassFractionalQuadSin import FractionalQuadSin


def multivar_gaussian_kernel(bw, x):
    D = x.shape[-1]
    inv_H = np.diag(np.power(bw, -2))
    norm_const = 1 / np.sqrt((2. * np.pi) ** D * (1. / np.linalg.det(inv_H)))
    exponent = -0.5 * np.einsum('...i,ij,...j', x, inv_H, x)
    return norm_const * np.exp(exponent)


def rmse_ignore_nans(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # Ignore NaNs in both arrays
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


# In[18]:


config = get_config()

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
fQuadSin = FractionalQuadSin(quad_coeff=config.quad_coeff, sin_coeff=config.sin_coeff,
                             sin_space_scale=config.sin_space_scale, diff=diff, X0=initial_state)
is_path_observations = np.array(
    [fQuadSin.euler_simulation(H=H, N=num_time_steps, deltaT=deltaT, isUnitInterval=isUnitInterval,
                               X0=initial_state, Ms=None, gaussRvs=rvs,
                               t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
    (num_paths, num_time_steps + 1))

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
    kernel_weights_unnorm = multivar_gaussian_kernel(bw=bw, x=prevPath_observations[:, :, np.newaxis, np.newaxis] - x[np.newaxis,np.newaxis, :, :])
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
    assert (denominator.shape == (x.shape[0], 1))
    numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, np.newaxis], axis=(1, 0)) / N * (
            t1 - t0)
    assert (numerator.shape == x.shape)
    estimator = numerator / denominator
    assert (estimator.shape == x.shape)
    # assert all([np.all(estimator[i, :] == numerator[i,:]/denominator[i,0]) for i in range(estimator.shape[0])])
    # This is the "truncated" discrete drift estimator to ensure appropriate risk bounds
    if truncate:
        m = np.min(denominator[:, 0])
        estimator[denominator[:, 0] <= m / 2., :] = 0.
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


bws = np.logspace(-4, -0.05, 20)


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -(4. * config.quartic_coeff * np.power(prev, 3) + 2. * config.quad_coeff * prev + config.const)
    return drift_X[:, np.newaxis, :]


num_time_steps = 100
num_state_paths = 10
for k in range(len(bws)):
    bw = bws[k]
    print(f"Considering bandwidth grid number {k}\n")
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
        global_mean = IID_NW_estimator(prevPath_observations=prevPath_observations, bw=np.array([bw]),
                                                x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                                t0=config.t0, truncate=True)[:, np.newaxis, :]
        local_mean = IID_NW_estimator(prevPath_observations=prevPath_observations, bw=np.array([bw]),
                                               x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=True)[:, np.newaxis, :]
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps

    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fQuadSinHF_DriftTracking_{round(bw, 4)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c").replace(
        ".", "")
    np.save(save_path + "_true_states.npy", true_states)
    np.save(save_path + "_global_states.npy", global_states)
    np.save(save_path + "_local_states.npy", local_states)

