#!/usr/bin/env python
# coding: utf-8
from multiprocessing import shared_memory

from configs import project_config
import multiprocessing as mp
import numpy as np
from configs.RecursiveVPSDE.Markovian_fSinLog.recursive_Markovian_PostMeanScore_fSinLog_HighFTh_T256_H05_tl_110data_StbleTgt_WRMSE import get_config
from src.classes.ClassFractionalSinLog import FractionalSinLog

from tqdm import tqdm

from utils.drift_evaluation_functions import process_single_R_hermite


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = (-np.sin(config.sin_space_scale*prev)*np.log(1+config.log_space_scale*np.abs(prev))/config.sin_space_scale)
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
        paths = np.load(config.data_path, allow_pickle=True)
        paths = np.concatenate(
            [np.repeat(np.array(config.initState).reshape((1, 1)), paths.shape[0], axis=0),
             paths], axis=1)
        print(paths.shape, num_paths, config.ts_length + 1)
        assert paths.shape == (num_paths, config.ts_length + 1)
    except (FileNotFoundError, AssertionError) as e:
        fSinLog = FractionalSinLog(log_space_scale=config.log_space_scale,
                                   sin_space_scale=config.sin_space_scale, diff=diff, X0=initial_state)
        paths = np.array(
            [fSinLog.euler_simulation(H=H, N=num_time_steps, deltaT=deltaT, isUnitInterval=isUnitInterval,
                                      X0=initial_state,
                                      Ms=None, gaussRvs=rvs,
                                      t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
            (num_paths, num_time_steps + 1))
        np.save(config.data_path, paths[:, 1:])
        assert paths.shape == (num_paths, config.ts_length + 1)

    prevPath_shm = shared_memory.SharedMemory(create=True, size=paths.nbytes)

    # Create numpy arrays from the shared memory buffers
    prevPath_shm_array = np.ndarray(paths.shape, dtype=np.float64, buffer=prevPath_shm.buf)

    # Copy the data into the shared memory arrays
    np.copyto(prevPath_shm_array, paths)

    num_time_steps = 100
    num_state_paths = 100
    rmse_quantile_nums = 20
    # Euler-Maruyama Scheme for Tracking Errors
    shape = paths.shape
    Rs = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    for R_idx in tqdm(range(Rs.shape[0])):
        R = Rs[R_idx]
        with mp.Pool(processes=rmse_quantile_nums) as pool:
            # Prepare the arguments for each task
            tasks = [(quant_idx, R, shape, true_drift, config, num_time_steps, num_state_paths, deltaT,
                      prevPath_shm.name) for quant_idx in range(rmse_quantile_nums)]

            # Run the tasks in parallel
            results = pool.starmap(process_single_R_hermite, tasks)
        results = {k: v for d in results for k, v in d.items()}
        all_true_states = np.concatenate([v[0][np.newaxis, :] for v in results.values()], axis=0)
        all_global_states = np.zeros(shape=(rmse_quantile_nums, num_state_paths, 1 + num_time_steps, config.ndims))
        all_local_states = np.concatenate([v[1][np.newaxis, :] for v in results.values()], axis=0)
        assert (all_true_states.shape == all_global_states.shape == all_local_states.shape)
        save_path = (
                project_config.ROOT_DIR + f"experiments/results/Hermite_fSinLog_DriftTrack_{R}R_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.log_space_scale}b_{config.sin_space_scale}c").replace(
            ".", "")
        print(f"Save path {save_path}\n")
        np.save(save_path + "_true_states.npy", all_true_states)
        np.save(save_path + "_global_states.npy", all_global_states)
        np.save(save_path + "_local_states.npy", all_local_states)


