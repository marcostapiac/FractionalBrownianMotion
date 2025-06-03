import numpy as np
from tqdm import tqdm

from configs import project_config
import multiprocessing as mp
from multiprocessing import shared_memory
from configs.RecursiveVPSDE.Markovian_fBiPotDDims.recursive_Markovian_PostMeanScore_fBiPot12Dims_T256_H05_tl_110data_StbleTgt_WRMSE import \
        get_config
from src.classes.ClassFractionalBiPotential import FractionalBiPotential
from utils.drift_evaluation_functions import IID_NW_multivar_estimator, process_IID_bandwidth


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = -(4. * np.array(config.quartic_coeff) * np.power(prev,
                                                               3) + 2. * np.array(config.quad_coeff) * prev + np.array(
        config.const))
    return drift_X[:, np.newaxis, :]

if __name__ == "__main__":
    config = get_config()
    num_paths = 10240
    t0 = config.t0
    deltaT = config.deltaT
    t1 = deltaT * config.ts_length
    # Drift parameters
    diff = config.diffusion
    initial_state = np.array(config.initState)
    rvs = None
    H = config.hurst
    try:
        is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :, :]
        is_path_observations = np.concatenate(
            [np.repeat(np.array(config.initState).reshape((1, 1, config.ndims)), is_path_observations.shape[0], axis=0),
             is_path_observations], axis=1)
        assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.ndims)
    except (FileNotFoundError, AssertionError) as e:
        print(e)
        fLnz = FractionalBiPotential(num_dims=config.ndims, const=config.const, quartic_coeff=config.quartic_coeff, quad_coeff=config.quad_coeff,
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
    prevPath_observations = path_observations[:, 1:-1, :]
    # We compute the path incs with respect to the prevPath_observations (since X_{t_{0}} != X_{0})
    path_incs = np.diff(path_observations, axis=1)[:, 1:, :]
    assert (prevPath_observations.shape == path_incs.shape)
    assert (path_incs.shape[1] == config.ts_length - 1)
    assert (path_observations.shape[1] == prevPath_observations.shape[1] + 2)
    assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))
    grid_1d = np.logspace(-4, -1, 40)
    bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
    assert (bws.shape == (40, config.ndims))

    prevPath_shm = shared_memory.SharedMemory(create=True, size=prevPath_observations.nbytes)
    path_incs_shm = shared_memory.SharedMemory(create=True, size=path_incs.nbytes)

    # Create numpy arrays from the shared memory buffers
    prevPath_shm_array = np.ndarray(prevPath_observations.shape, dtype=np.float64, buffer=prevPath_shm.buf)
    path_incs_shm_array = np.ndarray(path_incs.shape, dtype=np.float64, buffer=path_incs_shm.buf)

    # Copy the data into the shared memory arrays
    np.copyto(prevPath_shm_array, prevPath_observations)
    np.copyto(path_incs_shm_array, path_incs)

    num_time_steps = 256
    num_state_paths = 100
    rmse_quantile_nums = 10
    # Euler-Maruyama Scheme for Tracking Errors
    shape = prevPath_observations.shape
    for bw_idx in tqdm(range(bws.shape[0])):
        bw = bws[bw_idx,:]
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
                project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fBiPot_{config.ndims}DDims_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff[0]}a_{config.quad_coeff[0]}b_{config.const[0]}c").replace(
            ".", "")
        print(f"Save path {save_path}\n")
        np.save(save_path + "_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)