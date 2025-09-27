



import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
from tqdm import tqdm

from configs import project_config
from configs.RecursiveVPSDE.Markovian_StochasticBurgers.recursive_Markovian_PostMeanScore_ImagStochasticBurgers_T256_H05_tl_110data_StbleTgt import \
    get_config
from utils.drift_evaluation_functions import process_IID_bandwidth, stochastic_burgers_drift, \
    process_IID_SBurgers_bandwidth
from utils.resource_logger import ResourceLogger, set_runtime_global


if __name__ == "__main__":
    config = get_config()
    with ResourceLogger(
            interval=120,
            outfile=config.nadaraya_resource_logging_path,  # path where log will be written
            job_type="CPU multiprocessing drift evaluation",
    ):
        num_paths = 10240
        t0 = config.t0
        assert not config.real
        deltaT = config.deltaT
        t1 = deltaT * config.ts_length
        # Drift parameters
        diff = config.diffusion
        initial_state = np.array(config.initState)
        rvs = None
        H = config.hurst
        is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :, :,:]
        is_path_observations = is_path_observations[:, :, :, 0] if config.real else is_path_observations[:, :, :, 1]
        is_path_observations = np.concatenate(
            [np.repeat(np.array(config.initState).reshape((1, 1, config.num_dims)), is_path_observations.shape[0], axis=0),
             is_path_observations], axis=1)
        assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.num_dims)
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
        grid_1d = np.logspace(-3.55, -0.05, 30)
        bws = np.stack([grid_1d for m in range(config.num_dims)], axis=-1)
        assert (bws.shape == (30, config.num_dims))

        prevPath_shm = shared_memory.SharedMemory(create=True, size=prevPath_observations.nbytes)
        path_incs_shm = shared_memory.SharedMemory(create=True, size=path_incs.nbytes)

        # Create numpy arrays from the shared memory buffers
        prevPath_shm_array = np.ndarray(prevPath_observations.shape, dtype=np.complex128, buffer=prevPath_shm.buf)
        path_incs_shm_array = np.ndarray(path_incs.shape, dtype=np.complex128, buffer=path_incs_shm.buf)

        # Copy the data into the shared memory arrays
        np.copyto(prevPath_shm_array, prevPath_observations)
        np.copyto(path_incs_shm_array, path_incs)

        num_time_steps = 256
        num_state_paths = 100
        rmse_quantile_nums = 2
        # Ensure randomness across starmap calls
        master_seed = 42
        seed_seq = np.random.SeedSequence(master_seed)
        child_seeds = seed_seq.spawn(rmse_quantile_nums)  # One per quant_idx  # One per quant_idx

        # Euler-Maruyama Scheme for Tracking Errors
        shape = prevPath_observations.shape
        for bw_idx in tqdm(range(15,bws.shape[0])):
            set_runtime_global(idx=bw_idx)
            bw = bws[bw_idx, :]
            inv_H = np.diag(np.power(bw, -2))
            norm_const = 1 / np.sqrt((2. * np.pi) ** config.num_dims * (1. / np.linalg.det(inv_H)))

            print(f"Considering bandwidth grid number {bw_idx}\n")
            with mp.Pool(processes=rmse_quantile_nums) as pool:
                # Prepare the arguments for each task
                tasks = [(quant_idx, shape, inv_H, norm_const, stochastic_burgers_drift, config, num_time_steps, num_state_paths, deltaT,
                          prevPath_shm.name, path_incs_shm.name, child_seeds[quant_idx]) for quant_idx in
                         range(rmse_quantile_nums)]

                # Run the tasks in parallel
                results = pool.starmap(process_IID_SBurgers_bandwidth, tasks)
            results = {k: v for d in results for k, v in d.items()}
            all_true_states = np.concatenate([v[0][np.newaxis, :] for v in results.values()], axis=0)
            all_local_states = np.concatenate([v[2][np.newaxis, :] for v in results.values()], axis=0)
            all_global_states = np.concatenate([v[1][np.newaxis, :] for v in results.values()], axis=0)
            assert (all_true_states.shape == all_global_states.shape == all_local_states.shape)

            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_SBurgers_{config.num_dims}DDims_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.nu}nu_{config.alpha}alph").replace(
                ".", "")
            print(f"Save path {save_path}\n")
            np.save(save_path + "_true_states.npy", all_true_states)
            np.save(save_path + "_global_states.npy", all_global_states)
            np.save(save_path + "_local_states.npy", all_local_states)
