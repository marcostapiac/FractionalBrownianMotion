import numpy as np
from configs import project_config
from tqdm import tqdm
from scipy.stats import norm
from configs.RecursiveVPSDE.LSTM_fMullerBrown.recursive_LSTM_PostMeanScore_MullerBrown_T256_H05_tl_110data import \
    get_config
from src.classes.ClassFractionalMullerBrown import FractionalMullerBrown
from utils.drift_evaluation_functions import IID_NW_multivar_estimator

config = get_config()
num_paths = 109
t0 = config.t0
deltaT = config.deltaT
t1 = deltaT * config.ts_length
# Drift parameters
diff = config.diffusion
initial_state = np.array(config.initState)
rvs = None
H = config.hurst


fMB = FractionalMullerBrown(initialState=np.array(initial_state), X0s=np.array(config.X0s), Y0s=np.array(config.Y0s),
                            diff=config.diffusion, Aks=np.array(config.Aks),
                            aks=np.array(config.aks), bks=np.array(config.bks), cks=np.array(config.cks))
is_path_observations = np.array(
    [fMB.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, X0=initial_state, Ms=None, gaussRvs=rvs,
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
assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))

grid_1d = np.logspace(-4, -0.05, 40)
# mesh = np.meshgrid(*([grid_1d] * config.ndims), indexing='ij')
# Stack and reshape the grid so each row is a point in the n-dimensional grid
# bws = np.stack([m.ravel() for m in mesh], axis=-1)
bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
assert (bws.shape == (40, config.ndims))
assert (config.ndims == 2)


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    Aks = np.array(config.Aks)[np.newaxis, :]
    aks = np.array(config.aks)[np.newaxis, :]
    bks = np.array(config.bks)[np.newaxis, :]
    cks = np.array(config.cks)[np.newaxis, :]
    X0s = np.array(config.X0s)[np.newaxis, :]
    Y0s = np.array(config.Y0s)[np.newaxis, :]
    common = Aks * np.exp(aks* np.power(prev[:,[0]] - X0s, 2) \
                                 + bks* (prev[:,[0]] - X0s) * (prev[:, [1]] - Y0s)
                                 + cks* np.power(prev[:, [1]] - Y0s, 2))
    assert (common.shape == (num_paths, 4))
    drift_X = np.zeros((num_paths, config.ndims))
    drift_X[:, 0] = -np.sum(common * (2. * aks* (prev[:, [0]] - X0s) + bks* (prev[:, [1]] - Y0s)), axis=1)
    drift_X[:, 1] = -np.sum(common * (2. * cks* (prev[:, [1]] - Y0s) + bks* (prev[:, [0]] - X0s)), axis=1)

    return drift_X[:, np.newaxis, :]


num_time_steps = 100
num_state_paths = 100
rmse_quantile_nums = 20
# Euler-Maruyama Scheme for Tracking Errors
for k in range(bws.shape[0]):
    bw = bws[k, :]
    inv_H = np.diag(np.power(bw, -2))
    norm_const = 1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H)))
    print(f"Considering bandwidth grid number {k}\n")
    all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_global_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_local_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    for quant_idx in tqdm(range(rmse_quantile_nums)):
        true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
        # global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
        local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
        # Initialise the "true paths"
        true_states[:, [0], :] = config.initState
        # global_states[:, [0], :] = config.initState
        local_states[:, [0], :] = config.initState
        for i in tqdm(range(1, num_time_steps + 1)):
            eps = np.random.randn(num_state_paths, 1, config.ndims) * np.sqrt(deltaT)
            assert (eps.shape == (num_state_paths, 1, config.ndims))
            true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)
            # global_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, bw=bw,
            #                                        x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
            #                                        t0=config.t0, truncate=True)[:, np.newaxis, :]
            local_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H,
                                                   norm_const=norm_const,
                                                   x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                                   t0=config.t0, truncate=True)[:, np.newaxis, :]
            true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
            # global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
            local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
        all_true_states[quant_idx, :, :, :] = true_states
        # all_global_states[quant_idx, :, :, :] = global_states
        all_local_states[quant_idx, :, :, :] = local_states
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fMullerBrown_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT").replace(
        ".", "")
    print(f"Save path {save_path}\n")
    np.save(save_path + "_true_states.npy", all_true_states)
    np.save(save_path + "_global_states.npy", all_global_states)
    np.save(save_path + "_local_states.npy", all_local_states)

# In[10]:
