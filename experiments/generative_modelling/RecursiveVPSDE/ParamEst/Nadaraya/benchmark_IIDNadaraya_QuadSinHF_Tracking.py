import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from tqdm import tqdm
from configs import project_config
from configs.RecursiveVPSDE.LSTM_fQuadSinHF.recursive_LSTM_PostMeanScore_fQuadSinHF_T256_H05_tl_110data import \
    get_config
from src.classes.ClassFractionalQuadSin import FractionalQuadSin
from utils.drift_evaluation_functions import process_IID_bandwidth


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -2. * config.quad_coeff * prev + config.sin_coeff * config.sin_space_scale * np.sin(
        config.sin_space_scale * prev)
    return drift_X[:, np.newaxis, :]


if __name__ == "__main__":
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
    try:
        is_path_observations = np.load(config.data_path, allow_pickle=True)
        is_path_observations = np.concatenate(
            [np.repeat(np.array(config.initState).reshape((1, 1)), is_path_observations.shape[0], axis=0),
             is_path_observations], axis=1)
        assert is_path_observations.shape == (num_paths, config.ts_length + 1)
    except FileNotFoundError as e:

        fQuadSin = FractionalQuadSin(quad_coeff=config.quad_coeff, sin_coeff=config.sin_coeff,
                                     sin_space_scale=config.sin_space_scale, diff=diff, X0=initial_state)
        is_path_observations = np.array(
            [fQuadSin.euler_simulation(H=H, N=num_time_steps, deltaT=deltaT, isUnitInterval=isUnitInterval,
                                       X0=initial_state, Ms=None, gaussRvs=rvs,
                                       t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
            (num_paths, num_time_steps + 1))
        np.save(config.data_path, is_path_observations[:, 1:])

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
            tasks = [(quant_idx, shape, inv_H, norm_const, true_drift, config, num_time_steps, num_state_paths, deltaT,
                      prevPath_shm.name, path_incs_shm.name) for quant_idx in range(rmse_quantile_nums)]

            # Run the tasks in parallel
            results = pool.starmap(process_IID_bandwidth, tasks)
        results = {k: v for d in results for k, v in d.items()}
        all_true_states = np.concatenate([v[0][np.newaxis, :] for v in results.values()], axis=0)
        all_global_states = np.zeros(shape=(rmse_quantile_nums, num_state_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.concatenate([v[1][np.newaxis, :] for v in results.values()], axis=0)
        assert (all_true_states.shape == all_global_states.shape == all_local_states.shape)
        save_path = (
                project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fQuadSinHF_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.ts_length}NumDPS").replace(
            ".", "")
        np.save(save_path + "_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)
