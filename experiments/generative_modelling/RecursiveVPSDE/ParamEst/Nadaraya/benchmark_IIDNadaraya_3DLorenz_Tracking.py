import numpy as np
from configs import project_config
from tqdm import tqdm
from scipy.stats import norm
from configs.RecursiveVPSDE.LSTM_3DLorenz.recursive_LSTM_PostMeanScore_3DLorenz_T256_H05_tl_110data import get_config
from src.classes.ClassFractionalLorenz63 import FractionalLorenz63


def gaussian_kernel(bw, x):
    return norm.pdf(x / bw) / bw


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


# In[3]:


config = get_config()
num_paths = 10952
t0 = config.t0
deltaT = config.deltaT
t1 = deltaT * config.ts_length
# Drift parameters
diff = config.diffusion
initial_state = np.array(config.initState)
rvs = None
H = config.hurst

assert (config.ndims == 3)

fLnz = FractionalLorenz63(initialState=initial_state, diff=config.diffusion, sigma=config.ts_sigma, beta=config.ts_beta,
                          rho=config.ts_rho)
is_path_observations = np.array(
    [fLnz.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, X0=initial_state, Ms=None, gaussRvs=rvs,
                           t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
    (num_paths, config.ts_length + 1, config.ndims))

is_idxs = np.arange(is_path_observations.shape[0])
path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :, :]
# We note that we DO NOT evaluate the drift at time t_{0}=0
# We therefore remove the first element of path_observations since it includes X_{t_{0}} = X_{0}
# We also remove the last element since we never evaluate the drift at that point
t0 = deltaT
prevPath_observations = path_observations[:, 1:-1, :]
# We compute the path incs with respect to the prevPath_observations (since X_{t_{0}} != X_{0})
path_incs = np.diff(path_observations, axis=1)[:, 1:, :]
assert (prevPath_observations.shape == path_incs.shape)
assert (path_incs.shape[1] == config.ts_length - 1)
assert (path_observations.shape[1] == prevPath_observations.shape[1] + 2)


# In[5]:


def IID_NW_multivar_estimator(prevPath_observations, path_incs, bw, x, t1, t0, truncate):
    N, n, d = prevPath_observations.shape
    kernel_weights_unnorm = multivar_gaussian_kernel(bw=bw, x=prevPath_observations[:, :, np.newaxis, :] - x[np.newaxis,
                                                                                                           np.newaxis,
                                                                                                           :, :])
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
    assert (denominator.shape == (x.shape[0], 1))
    numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, :], axis=(1, 0)) / N * (
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

grid_1d = np.logspace(-4, -0.05, 40)
# mesh = np.meshgrid(*([grid_1d] * config.ndims), indexing='ij')
# Stack and reshape the grid so each row is a point in the n-dimensional grid
# bws = np.stack([m.ravel() for m in mesh], axis=-1)
bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
print(bws.shape)


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = np.zeros((num_paths, config.ndims))
    drift_X[:, 0] = config.ts_sigma * (prev[:, 1] - prev[:, 0])
    drift_X[:, 1] = (prev[:, 0] * (config.ts_rho - prev[:, 2]) - prev[:, 1])
    drift_X[:, 2] = (prev[:, 0] * prev[:, 1] - config.ts_beta * prev[:, 2])
    return drift_X[:, np.newaxis, :]


num_time_steps = 100
num_state_paths = 10
# Euler-Maruyama Scheme for Tracking Errors
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
        global_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, bw=bw,
                                                x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                                t0=config.t0, truncate=True)[:, np.newaxis, :]
        local_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, bw=bw,
                                               x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=True)[:, np.newaxis, :]
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_f{config.ndims}DLnz_DriftTrack_{bw[0]}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.ts_beta:.1e}Beta_{config.ts_rho:.1e}Rho_{config.ts_sigma:.1e}Sigma").replace(
        ".", "")
    print(f"Save path {save_path}\n")
    np.save(save_path + "_true_states.npy", true_states)
    np.save(save_path + "_global_states.npy", global_states)
    np.save(save_path + "_local_states.npy", local_states)

# In[10]:
