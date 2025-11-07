
from multiprocessing import shared_memory

import numpy as np
from tqdm import tqdm

from configs import project_config
from configs.RecursiveVPSDE.Markovian_12DLorenz.recursive_Markovian_PostMeanScore_12DLorenz_Stable_T256_H05_tl_110data_StbleTgt import \
    get_config
from src.classes.ClassFractionalLorenz96 import FractionalLorenz96
from utils.resource_logger import ResourceLogger, set_runtime_global

import math
import numpy as np
import torch


# ---------------------------
#   Helper: device + dtypes
# ---------------------------

def _get_device(device_str: str | None = None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------
#   Drift (Lorenz96) — vectorized on LongerTimes_GPU
# -----------------------------------------

@torch.no_grad()
def true_drift_gpu(prev: torch.Tensor, num_paths: int, config) -> torch.Tensor:
    """
    prev: (num_paths, d) float32 CUDA tensor
    returns: (num_paths, 1, d) float32 CUDA tensor
    """
    # Ensure shape
    assert prev.ndim == 2 and prev.shape[0] == num_paths and prev.shape[1] == config.ndims
    x = prev
    # Vectorized Lorenz96 drift:
    x_ip1 = torch.roll(x, shifts=-1, dims=1)  # x_{i+1}
    x_im1 = torch.roll(x, shifts=1,  dims=1)  # x_{i-1}
    x_im2 = torch.roll(x, shifts=2,  dims=1)  # x_{i-2}
    drift = (x_ip1 - x_im2) * x_im1 - x*float(config.forcing_const)
    return drift[:, None, :]


# ------------------------------------------------------------
#   IID Nadaraya–Watson multivariate estimator — LongerTimes_GPU & tiled
#   (No (N,n,M,d) materialization; stable; diagonal A fastpath)
# ------------------------------------------------------------

@torch.no_grad()
def IID_NW_multivar_estimator_gpu(
    prevPath_observations: torch.Tensor,  # (N,n,d) float32 CUDA
    path_incs: torch.Tensor,              # (N,n,d) float32 CUDA
    inv_H: torch.Tensor,                  # (d,) (diag) or (d,d) float32 CUDA
    norm_const: float,                    # same meaning as your CPU code
    x: torch.Tensor,                      # (M,d) float32 CUDA
    t1: float,
    t0: float,
    truncate: bool = True,
    M_tile: int = 32,                     # micro-batch states
    Nn_tile: int | None = 512_000,        # micro-batch samples (None => full)
    stable: bool = True,
) -> torch.Tensor:
    """
    Returns: (M,d) float32 CUDA tensor (keeps all heavy ops on LongerTimes_GPU).
    Matches your scaling:
      denom = sum(w)/(N*n)
      numer = (sum(w * incs)/N) * (t1 - t0)
    """
    #assert prevPath_observations.is_cuda and path_incs.is_cuda and x.is_cuda
    assert prevPath_observations.dtype == torch.float32
    assert path_incs.dtype == torch.float32
    assert x.dtype == torch.float32

    N, n, d = prevPath_observations.shape
    Nn = N * n
    if Nn_tile is None or Nn_tile > Nn:
        Nn_tile = Nn

    # Flatten once
    mu = prevPath_observations.reshape(Nn, d).contiguous()  # (Nn,d)
    dX = path_incs.reshape(Nn, d).contiguous()              # (Nn,d)

    # Diagonal vs full inv_H
    diag = (inv_H.ndim == 1)
    if diag:
        A = inv_H                                           # (d,)
        muAh = mu * A                                       # (Nn,d)
        mu_quad = (mu * muAh).sum(-1)                       # (Nn,)
        def xAh(X): return X * A
    else:
        A = inv_H                                           # (d,d)
        muAh = mu @ A                                       # (Nn,d)
        mu_quad = (mu * muAh).sum(-1)                       # (Nn,)
        # Sanity: PD
        sign, _ = torch.linalg.slogdet(A)
        if sign.item() <= 0:
            raise ValueError("inv_H must be positive definite.")

    # Use log(norm_const) directly to match your CPU estimator
    log_norm_const = float(np.log(norm_const))

    M = x.size(0)
    # Accumulate in float64 for stability
    denom = torch.zeros(M,     dtype=torch.float64, device=x.device)
    numer = torch.zeros(M, d,  dtype=torch.float64, device=x.device)

    for m0 in range(0, M, M_tile):
        X = x[m0:m0 + M_tile]                     # (mb,d)
        XAh = xAh(X)                               # (mb,d)
        X_quad = (X * XAh).sum(-1)                 # (mb,)

        denom_tile = torch.zeros(X.size(0),    dtype=torch.float64, device=x.device)
        numer_tile = torch.zeros(X.size(0), d, dtype=torch.float64, device=x.device)

        if stable:
            lse_max = torch.full((X.size(0),), -torch.inf, dtype=torch.float32, device=x.device)
            # First pass: find max exponent per state (over all Nn tiles)
            for i0 in range(0, Nn, Nn_tile):
                muq_i  = mu_quad[i0:i0 + Nn_tile]             # (bn,)
                muAh_i = muAh[i0:i0 + Nn_tile]                # (bn,d)
                cross  = muAh_i @ X.t()                       # (bn,mb)
                expo   = log_norm_const - 0.5 * (muq_i[:, None] + X_quad[None, :] - 2.0 * cross)
                lse_max = torch.maximum(lse_max, expo.max(dim=0).values)

        # Second pass: accumulate with stabilization (or plain)
        for i0 in range(0, Nn, Nn_tile):
            muAh_i = muAh[i0:i0 + Nn_tile]                    # (bn,d)
            muq_i  = mu_quad[i0:i0 + Nn_tile]                 # (bn,)
            dX_i   = dX[i0:i0 + Nn_tile]                      # (bn,d)

            cross = muAh_i @ X.t()                            # (bn,mb)
            expo  = log_norm_const - 0.5 * (muq_i[:, None] + X_quad[None, :] - 2.0 * cross)

            if stable:
                w = torch.exp(expo - lse_max[None, :])        # (bn,mb)
            else:
                w = torch.exp(expo)

            denom_tile += w.sum(dim=0, dtype=torch.float64) / (N * n)
            numer_tile += (w.t() @ dX_i).to(torch.float64) * ((t1 - t0) / N)

        if stable:
            scale = torch.exp(lse_max.to(torch.float64))
            denom_tile *= scale
            numer_tile *= scale[:, None]

        denom[m0:m0 + X.size(0)] += denom_tile
        numer[m0:m0 + X.size(0)] += numer_tile

    est = (numer / denom[:, None]).to(torch.float32)          # (M,d)

    if truncate:
        m = denom.min()
        est[denom <= (m / 2.0)] = 0

    return est


# ---------------------------------------------------------
#   End-to-end single-quantile simulator — LongerTimes_GPU & tiled
# ---------------------------------------------------------

@torch.no_grad()
def process_IID_bandwidth_gpu(
    quant_idx: int,
    shape: tuple,                 # (N, n, d)
    inv_H_np: np.ndarray,         # (d,d) diag matrix or (d,) diag vector
    norm_const: float,
    config,                       # your config object
    num_time_steps: int,
    num_state_paths: int,
    deltaT: float,
    prevPath_np: np.ndarray,      # (N, n, d) float64/32 — host
    path_incs_np: np.ndarray,     # (N, n, d) float64/32 — host
    seed_seq,                     # numpy.SeedSequence child
    device_str: str | None = None,
    M_tile: int = 32,
    Nn_tile: int | None = 512_000,
    stable: bool = True,
):
    """
    Mirrors your original `process_IID_bandwidth`, but fully on LongerTimes_GPU, and returns:
      { quant_idx: (true_states, global_states, local_states) }  as numpy arrays
    """
    device = _get_device(device_str)

    # Upload once (float32)
    prevPath = torch.as_tensor(prevPath_np, dtype=torch.float32, device=device).contiguous()
    path_incs = torch.as_tensor(path_incs_np, dtype=torch.float32, device=device).contiguous()

    # inv_H: prefer diagonal vector if possible
    inv_H_np = np.asarray(inv_H_np)
    if inv_H_np.ndim == 2 and np.allclose(inv_H_np, np.diag(np.diag(inv_H_np))):
        inv_H_vec = np.diag(inv_H_np).astype(np.float32)
        inv_H = torch.as_tensor(inv_H_vec, device=device)
    else:
        inv_H = torch.as_tensor(inv_H_np.astype(np.float32), device=device)

    d = config.ndims
    # States on LongerTimes_GPU
    true_states   = torch.zeros((num_state_paths, 1 + num_time_steps, d), dtype=torch.float32, device=device)
    global_states = torch.zeros_like(true_states)
    local_states  = torch.zeros_like(true_states)

    init = torch.as_tensor(np.asarray(config.initState, dtype=np.float32), device=device)
    true_states[:, 0, :]   = init
    global_states[:, 0, :] = init
    local_states[:, 0, :]  = init

    # RNG on LongerTimes_GPU (deterministic from provided SeedSequence)
    g = torch.Generator(device=device)
    seed = int(seed_seq.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))
    g.manual_seed(seed)

    for i in range(1, num_time_steps + 1):
        eps = torch.randn((num_state_paths, d), generator=g, device=device, dtype=torch.float32)
        eps *= (math.sqrt(deltaT) * float(config.diffusion))

        # True drift (LongerTimes_GPU)
        prev_true = true_states[:, i - 1, :]
        true_mean = true_drift_gpu(prev_true, num_state_paths, config)[:, 0, :]  # (M,d)

        # Global mean (at global state)
        x_global = global_states[:, i - 1, :]
        global_mean = IID_NW_multivar_estimator_gpu(
            prevPath, path_incs, inv_H, float(norm_const),
            x_global, float(config.t1), float(config.t0),
            truncate=True, M_tile=M_tile, Nn_tile=Nn_tile, stable=stable
        )

        # Local mean (at true state)
        x_true = true_states[:, i - 1, :]
        local_mean = IID_NW_multivar_estimator_gpu(
            prevPath, path_incs, inv_H, float(norm_const),
            x_true, float(config.t1), float(config.t0),
            truncate=True, M_tile=M_tile, Nn_tile=Nn_tile, stable=stable
        )

        # Euler–Maruyama updates
        true_states[:,   i, :] = true_states[:,   i - 1, :] + true_mean  * deltaT + eps
        global_states[:, i, :] = global_states[:, i - 1, :] + global_mean * deltaT + eps
        local_states[:,  i, :] = true_states[:,   i - 1, :] + local_mean  * deltaT + eps

    # Return as numpy for saving consistency with your code
    return {
        quant_idx: (
            true_states.detach().cpu().numpy(),
            global_states.detach().cpu().numpy(),
            local_states.detach().cpu().numpy(),
        )
    }

if __name__ == "__main__":
    config = get_config()
    num_paths = 1024 if config.feat_thresh == 1. else 10240
    with ResourceLogger(
            interval=120,
            outfile=config.nadaraya_resource_logging_path.replace(".json.json", "_GPUNADARAYA.json.json"),  # path where log will be written
            job_type="GPU training",
    ):
        assert num_paths == 1024
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
            fLnz = FractionalLorenz96(X0=config.initState, diff=config.diffusion, num_dims=config.ndims,
                                      forcing_const=config.forcing_const)
            is_path_observations = np.array(
                [fLnz.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, X0=initial_state, Ms=None, gaussRvs=rvs,
                                       t0=t0, t1=t1) for _ in (range(num_paths))]).reshape(
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
        grid_1d = np.logspace(-3.55, -0.05, 30)
        xadd = np.logspace(-0.05, 1.0, 11)[1:]  # 10 values > -0.05
        grid_1d = np.concatenate([grid_1d, xadd])
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

        num_time_steps = int(1*config.ts_length)
        rmse_quantile_nums = 1
        # Ensure randomness across starmap calls
        master_seed = 42
        seed_seq = np.random.SeedSequence(master_seed)
        child_seeds = seed_seq.spawn(rmse_quantile_nums)  # One per quant_idx  # One per quant_idx

        # Euler-Maruyama Scheme for Tracking Errors
        shape = prevPath_observations.shape
        num_state_paths = 100
        mses = {bw_idx: np.inf for bw_idx in (range(10, bws.shape[0]))}
        for bw_idx in tqdm(range(10,bws.shape[0])):
            set_runtime_global(idx=bw_idx)
            bw = bws[bw_idx, :]
            inv_H = np.diag(np.power(bw, -2))
            norm_const = 1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H)))
            quant_idx = 0
            print(f"Considering bandwidth grid number {bw_idx}\n")
            results = {}
            out = process_IID_bandwidth_gpu(
                quant_idx=quant_idx,
                shape=prevPath_observations.shape,
                inv_H_np=inv_H,  # pass diag matrix or vector
                norm_const=norm_const,
                config=config,
                num_time_steps=num_time_steps,
                num_state_paths=num_state_paths,
                deltaT=deltaT,
                prevPath_np=prevPath_observations,
                path_incs_np=path_incs,
                seed_seq=child_seeds[quant_idx],
                device_str=None,  # or leave None to auto-pick
                M_tile=32,  # tune up/down
                Nn_tile=512_000,  # tune up/down (or None)
                stable=True,
            )
            results.update(out)

            # Then concatenate & save like before
            all_true_states = np.concatenate([v[0][np.newaxis, :] for v in results.values()], axis=0)
            all_global_states = np.concatenate([v[1][np.newaxis, :] for v in results.values()], axis=0)
            all_local_states = np.concatenate([v[2][np.newaxis, :] for v in results.values()], axis=0)
            assert (all_true_states.shape == all_global_states.shape == all_local_states.shape)
            assert num_paths == 1024

            save_path = (
                    project_config.ROOT_DIR + f"experiments/results/IIDNadarayaGPU_f{config.ndims}DLnz_DriftTrack_{round(bw[0], 6)}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.forcing_const}FConst").replace(
                ".", "")
            print(f"Save path for Track {save_path}\n")
            np.save(save_path + "_true_states.npy", all_true_states)
            np.save(save_path + "_global_states.npy", all_global_states)
            np.save(save_path + "_local_states.npy", all_local_states)

            M_tile = 1024
            Nn_tile = 512000
            stable = True
            num_dhats = 1 # No variability given we use same training dataset
            device = _get_device(None)
            all_true_states = all_true_states[np.random.choice(np.arange(all_true_states.shape[0]), 100), :, :]
            all_true_states = all_true_states.reshape(-1, config.ts_dims)
            unif_is_drift_hats = np.zeros((all_true_states.shape[0], num_dhats, config.ts_dims))
            Xs = torch.as_tensor(all_true_states, dtype=torch.float32, device=device).contiguous()
            for k in tqdm(range(num_dhats)):
                is_ss_path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False),
                                          :]
                is_prevPath_observations = is_ss_path_observations[:, 1:-1]
                is_path_incs = np.diff(is_ss_path_observations, axis=1)[:, 1:]
                is_prevPath_observations = torch.as_tensor(is_prevPath_observations, dtype=torch.float32,
                                                           device=device).contiguous()
                is_path_incs = torch.as_tensor(is_path_incs, dtype=torch.float32, device=device).contiguous()
                # inv_H: prefer diagonal vector if possible
                inv_H_np = np.asarray(inv_H)
                if inv_H_np.ndim == 2 and np.allclose(inv_H_np, np.diag(np.diag(inv_H_np))):
                    inv_H_vec = np.diag(inv_H_np).astype(np.float32)
                    inv_H = torch.as_tensor(inv_H_vec, device=device)
                else:
                    inv_H = torch.as_tensor(inv_H_np.astype(np.float32), device=device)

                unif_is_drift_hats[:, k, :] = IID_NW_multivar_estimator_gpu(
                    is_prevPath_observations, is_path_incs, inv_H, float(norm_const),
                    Xs, float(config.t1), float(config.t0),
                    truncate=True, M_tile=M_tile, Nn_tile=Nn_tile, stable=stable
                ).cpu().numpy()
            est_unif_is_drift_hats = np.mean(unif_is_drift_hats, axis=1)
            mses[bw_idx] = (
            bws[bw_idx], np.mean(np.sum(np.power(est_unif_is_drift_hats - all_true_states, 2), axis=-1), axis=-1))
            save_path = save_path.replace("DriftTrack", "DriftEvalExp")
            print(f"Save path for EvalExp {save_path}\n")
            # np.save(save_path + "_muhats_true_states.npy", all_true_states)
            # np.save(save_path + "_muhats.npy", unif_is_drift_hats)
        save_path = (
                project_config.ROOT_DIR + f"experiments/results/IIDNadarayaGPU_f{config.ndims}DLnz_DriftEvalExp_MSEs_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT_{config.forcing_const}FConst").replace(
            ".", "")
        np.savez(save_path, mses, allow_pickle=True)