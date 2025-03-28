import numpy as np
from tqdm import tqdm

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fBiPot.recursive_LSTM_PostMeanScore_fBiPot_T256_H05_tl_110data import get_config
from src.classes.ClassFractionalBiPotential import FractionalBiPotential
from utils.drift_evaluation_functions import IID_NW_multivar_estimator
import multiprocessing as mp
from multiprocessing import shared_memory


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -(4. * config.quartic_coeff * np.power(prev, 3) + 2. * config.quad_coeff * prev + config.const)
    return drift_X[:, np.newaxis, :]


def process_bandwidth(bw_idx, quant_idx, shape, inv_H, norm_const,config, num_time_steps, num_state_paths, deltaT, prevPath_name, path_incs_name):
    print(f"Bandwidth brid number {bw_idx} and quant idx {quant_idx}\n")
    # Attach to the shared memory blocks by name.
    shm_prev = shared_memory.SharedMemory(name=prevPath_name)
    shm_incs = shared_memory.SharedMemory(name=path_incs_name)

    # Create numpy array views from the shared memory (no copying happens here)
    prevPath_observations = np.ndarray(shape, dtype=np.float64, buffer=shm_prev.buf)
    path_incs = np.ndarray(shape, dtype=np.float64, buffer=shm_incs.buf)

    true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    # global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    # Initialise the "true paths"
    true_states[:, [0], :] = config.initState
    # global_states[:, [0], :] = config.initState
    local_states[:, [0], :] = config.initState
    for i in range(1, num_time_steps + 1):
        eps = np.random.randn(num_state_paths, 1, config.ndims) * np.sqrt(deltaT)
        assert (eps.shape == (num_state_paths, 1, config.ndims))
        true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)
        # global_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H, norm_const=norm_const,
        #                                       x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
        #                                       t0=config.t0, truncate=True)[:, np.newaxis, :]
        local_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H,
                                               norm_const=norm_const,
                                               x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=True)[:, np.newaxis, :]
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        # global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
    return {quant_idx: (true_states, local_states)}


if __name__ == '__main__':
    mp.set_start_method("spawn")
    config = get_config()
    num_paths = 10952
    num_time_steps = config.ts_length
    isUnitInterval = True
    diff = config.diffusion
    initial_state = config.initState
    rvs = None
    H = config.hurst
    deltaT = config.deltaT
    t0 = config.t0
    t1 = deltaT * num_time_steps
    fBiPot = FractionalBiPotential(const=config.const, quartic_coeff=config.quartic_coeff, quad_coeff=config.quad_coeff,
                                   diff=diff, X0=initial_state)
    is_path_observations = np.array(
        [fBiPot.euler_simulation(H=H, N=num_time_steps, deltaT=deltaT, isUnitInterval=isUnitInterval, X0=initial_state,
                                 Ms=None, gaussRvs=rvs,
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
    assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))
    bws = np.logspace(-4, -0.05, 40)

    prevPath_shm = shared_memory.SharedMemory(create=True, size=prevPath_observations.nbytes)
    path_incs_shm = shared_memory.SharedMemory(create=True, size=path_incs.nbytes)

    # Create numpy arrays from the shared memory buffers
    prevPath_shm_array = np.ndarray(prevPath_observations.shape, dtype=np.float64, buffer=prevPath_shm.buf)
    path_incs_shm_array = np.ndarray(path_incs.shape, dtype=np.float64, buffer=path_incs_shm.buf)

    # Copy the data into the shared memory arrays
    np.copyto(prevPath_shm_array, prevPath_observations)
    np.copyto(path_incs_shm_array, path_incs)

    num_time_steps = 100
    num_state_paths = 100
    rmse_quantile_nums = 20
    # Euler-Maruyama Scheme for Tracking Errors
    shape = prevPath_observations.shape
    for bw_idx in tqdm(range(bws.shape[0])):
        bw = np.array([bws[bw_idx]])
        inv_H = np.diag(np.power(bw, -2))
        norm_const = 1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H)))

        print(f"Considering bandwidth grid number {bw_idx}\n")
        with mp.Pool(processes=rmse_quantile_nums) as pool:
            # Prepare the arguments for each task
            tasks = [(bw_idx, quant_idx, shape, inv_H, norm_const, config, num_time_steps, num_state_paths, deltaT,
                      prevPath_shm.name, path_incs_shm.name) for quant_idx in range(rmse_quantile_nums)]

            # Run the tasks in parallel
            results = pool.starmap(process_bandwidth, tasks)
        results = {k: v for d in results for k, v in d.items()}
        all_true_states = np.concatenate([v[0][np.newaxis, :] for v in results.values()], axis=0)
        all_global_states = np.zeros(shape=(rmse_quantile_nums, num_state_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.concatenate([v[1][np.newaxis, :] for v in results.values()], axis=0)
        assert (all_true_states.shape == all_global_states.shape == all_local_states.shape)
        save_path = (
                project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fBiPot_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c").replace(
            ".", "")
        print(all_true_states.shape, all_global_states.shape, all_local_states.shape)
        np.save(save_path + "_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)

    # Cleanup shared memory
    prevPath_shm.close()
    path_incs_shm.close()
    prevPath_shm.unlink()
    path_incs_shm.unlink()
