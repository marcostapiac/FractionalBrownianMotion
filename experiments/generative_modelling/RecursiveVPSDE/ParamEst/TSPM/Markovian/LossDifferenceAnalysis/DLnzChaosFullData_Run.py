#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import io
import os
import torch
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalMarkovianTSPostMeanScoreMatching import \
    ConditionalMarkovianTSPostMeanScoreMatching
from utils.drift_evaluation_functions import experiment_MLP_DDims_drifts
from configs.RecursiveVPSDE.Markovian_8DLorenz.recursive_Markovian_PostMeanScore_8DLorenz_Chaos_T256_H05_tl_110data_StbleTgt_FULLDATA import get_config as get_8dlnz_config
from configs.RecursiveVPSDE.Markovian_12DLorenz.recursive_Markovian_PostMeanScore_12DLorenz_Chaos_T256_H05_tl_110data_StbleTgt_FULLDATA import get_config as get_12dlnz_config
from configs.RecursiveVPSDE.Markovian_20DLorenz.recursive_Markovian_PostMeanScore_20DLorenz_Chaos_T256_H05_tl_110data_StbleTgt_FULLDATA import get_config as get_20dlnz_config
from configs.RecursiveVPSDE.Markovian_40DLorenz.recursive_Markovian_PostMeanScore_40DLorenz_Chaos_T256_H05_tl_110data_StbleTgt_FULLDATA import get_config as get_40dlnz_config
from tqdm import tqdm
from utils.drift_evaluation_functions import multivar_score_based_MLP_drift_OOS
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from configs import project_config


# In[8]:


def _get_device(device_str: str | None = None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def true_drifts(device_id, config, state):
    true_drifts = np.zeros_like(state)
    for i in range(config.ndims):
        true_drifts[:, i] = (state[:, (i + 1) % config.ndims] - state[:, i - 2]) * state[:, i - 1] - state[:,i] * config.forcing_const
    return torch.tensor(true_drifts[:, np.newaxis, :], device=device_id, dtype=torch.float32)


# In[9]:


lnz_8d_config = get_8dlnz_config()
lnz_12d_config = get_12dlnz_config()
lnz_20d_config = get_20dlnz_config()
lnz_40d_config = get_40dlnz_config()
device_id = _get_device()
assert lnz_8d_config.feat_thresh == lnz_12d_config.feat_thresh != 1
assert lnz_20d_config.feat_thresh == lnz_40d_config.feat_thresh != 1
num_paths = 1024 if lnz_8d_config.feat_thresh == 1. else 10240
assert num_paths == 10240
root_dir ="/Users/marcos/Library/CloudStorage/OneDrive-ImperialCollegeLondon/StatML_CDT/Year2/DiffusionModels/"


# In[10]:


def generate_synthetic_paths(config, device_id, good, inv_H, norm_const, prevPath_observations, prevPath_incs, M_tile, Nn_tile, stable):
    # Prepare for Nadaraya
    inv_H_np = np.asarray(inv_H)
    if inv_H_np.ndim == 2 and np.allclose(inv_H_np, np.diag(np.diag(inv_H_np))):
        inv_H_vec = np.diag(inv_H_np).astype(np.float32)
        inv_H = torch.as_tensor(inv_H_vec, device=device_id)
    else:
        inv_H = torch.as_tensor(inv_H_np.astype(np.float32), device=device_id)
    prevPath_observations = torch.as_tensor(prevPath_observations, dtype=torch.float32,
                                               device=device_id).contiguous()
    prevPath_incs = torch.as_tensor(prevPath_incs, dtype=torch.float32, device=device_id).contiguous()
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    num_diff_times = 1
    rmse_quantile_nums = 1
    num_paths = 1000
    num_time_steps = int(5*config.ts_length)
    deltaT = config.deltaT
    all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_score_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_nad_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    for quant_idx in (range(rmse_quantile_nums)):
        good.eval()
        initial_state = np.repeat(np.atleast_2d(config.initState)[np.newaxis, :], num_paths, axis=0)
        assert (initial_state.shape == (num_paths, 1, config.ndims))

        true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        score_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        nad_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

        # Initialise the "true paths"
        true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
        # Initialise the "global score-based drift paths"
        score_states[:, [0], :] = true_states[:, [0], :]
        nad_states[:, [0], :] = true_states[:, [0],
                                  :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)

        # Euler-Maruyama Scheme for Tracking Errors
        for i in tqdm(range(1, num_time_steps + 1)):
            eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT) * config.diffusion

            assert (eps.shape == (num_paths, 1, config.ndims))
            true_mean = true_drifts(state=true_states[:, i - 1, :], device_id=device_id,config=config).cpu().numpy()
            denom = 1.
            score_mean = multivar_score_based_MLP_drift_OOS(score_model=good,
                                                             num_diff_times=num_diff_times,
                                                             diffusion=diffusion,
                                                             num_paths=num_paths,
                                                             ts_step=deltaT, config=config,
                                                             device=device_id,
                                                             prev=score_states[:, i - 1, :])

            x = torch.as_tensor(nad_states[:, i - 1, :], device=device_id, dtype=torch.float32).contiguous()
            nad_mean = IID_NW_multivar_estimator_gpu(
                        prevPath_observations=prevPath_observations, path_incs=prevPath_incs, inv_H=inv_H, norm_const=float(norm_const),
                        x=x, t1=float(config.t1), t0=float(config.t0),
                        truncate=False, M_tile=M_tile, Nn_tile=Nn_tile, stable=stable
                    ).cpu().numpy()[:, np.newaxis, :]
            true_states[:, [i], :] = (true_states[:, [i - 1], :] \
                                      + true_mean * deltaT \
                                      + eps) / denom
            score_states[:, [i], :] = (score_states[:, [i - 1], :] + score_mean * deltaT + eps) / denom
            nad_states[:, [i], :] = (nad_states[:, [i - 1], :] + nad_mean * deltaT + eps) / denom

        all_true_states[quant_idx, :, :, :] = true_states
        all_score_states[quant_idx, :, :, :] = score_states
        all_nad_states[quant_idx, :, :, :] = nad_states
    del prevPath_observations, prevPath_incs
    return all_true_states, all_score_states, all_nad_states, num_time_steps


def get_best_epoch(config, type):
    model_dir = "/".join(config.scoreNet_trained_path.split("/")[:-1]) + "/"
    for file in os.listdir(model_dir):
        if config.scoreNet_trained_path in os.path.join(model_dir, file) and f"{type}" in file:
            best_epoch = int(file.split(f"{type}NEp")[-1])
    return best_epoch

def get_best_track_file(root_score_dir, ts_type, best_epoch_track):
    for file in os.listdir(root_score_dir):
        if ("_"+str(best_epoch_track)+"Nep") in file and "true" in file and ts_type in file and "1000FTh" in file and "125FConst" in file:
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            true_file = np.load(root_score_dir+file, allow_pickle=True)
        elif ("_"+str(best_epoch_track)+"Nep") in file and "global" in file and ts_type in file and "1000FTh" in file and "125FConst" in file:
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            global_file = np.load(root_score_dir+file, allow_pickle=True)
    print(ts_type)
    return true_file, global_file

def get_best_eval_exp_file(config, root_score_dir, ts_type):
    best_epoch_eval = get_best_epoch(config=config,type="EE")
    for file in os.listdir(root_score_dir):
        if ("_"+str(best_epoch_eval)+"Nep") in file and "MSE" in file and ts_type in file and "1000FTh" in file and "125FConst" in file:
            print(f"Starting {file}\n")
            with open(root_score_dir+file, 'rb') as f:
                buf = io.BytesIO(f.read())  # hydrates once, sequentially
            print(f"Starting {file}\n")
            mse = pd.read_parquet(root_score_dir+file, engine="fastparquet")
    return mse
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
    Returns: (M,d) float32 CUDA tensor (keeps all heavy ops on Times_GPU).
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
            numer_tile += (w.t() @ dX_i).to(torch.float64) / (N * (t1 - t0))

        

        denom[m0:m0 + X.size(0)] += denom_tile
        numer[m0:m0 + X.size(0)] += numer_tile

    est = torch.full((M, d), float('nan'), dtype=torch.float32, device=x.device)
    mask = (denom > 0) & torch.isfinite(denom) & torch.isfinite(numer).all(dim=1)
    est[mask] = (numer[mask] / denom[mask, None]).to(torch.float32)

    return est
def prepare_for_nadaraya(config, num_paths):
    deltaT = config.deltaT
    t1 = deltaT * config.ts_length
    is_path_observations = np.load(config.data_path, allow_pickle=True)[:num_paths, :, :]
    is_path_observations = np.concatenate(
        [np.repeat(np.array(config.initState).reshape((1, 1, config.ndims)), is_path_observations.shape[0], axis=0),
         is_path_observations], axis=1)
    assert is_path_observations.shape == (num_paths, config.ts_length + 1, config.ndims)
    path_observations = is_path_observations
    t0 = deltaT
    prevPath_observations = path_observations[:, 1:-1, :]
    path_incs = np.diff(path_observations, axis=1)[:, 1:, :]
    assert (prevPath_observations.shape == path_incs.shape)
    assert (path_incs.shape[1] == config.ts_length - 1)
    assert (path_observations.shape[1] == prevPath_observations.shape[1] + 2)
    assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))
    prevPath_observations = path_observations[:, 1:-1, :]
    path_incs = np.diff(path_observations, axis=1)[:, 1:, :]
    return is_path_observations, prevPath_observations, path_incs

def run_nadaraya_single_bw(config, is_path_observations, states, M_tile, inv_H, norm_const, Nn_tile, stable):
    Xs = torch.as_tensor(states, dtype=torch.float32, device=device_id).contiguous()
    is_ss_path_observations = is_path_observations
    is_prevPath_observations = is_ss_path_observations[:, 1:-1]
    is_path_incs = np.diff(is_ss_path_observations, axis=1)[:, 1:]
    is_prevPath_observations = torch.as_tensor(is_prevPath_observations, dtype=torch.float32,
                                               device=device_id).contiguous()
    is_path_incs = torch.as_tensor(is_path_incs, dtype=torch.float32, device=device_id).contiguous()
    # inv_H: prefer diagonal vector if possible
    inv_H_np = np.asarray(inv_H)
    if inv_H_np.ndim == 2 and np.allclose(inv_H_np, np.diag(np.diag(inv_H_np))):
        inv_H_vec = np.diag(inv_H_np).astype(np.float32)
        inv_H = torch.as_tensor(inv_H_vec, device=device_id)
    else:
        inv_H = torch.as_tensor(inv_H_np.astype(np.float32), device=device_id)

    unif_is_drift_hats = IID_NW_multivar_estimator_gpu(
        is_prevPath_observations, is_path_incs, inv_H, float(norm_const),
        Xs, float(config.t1), float(config.t0),
        truncate=False, M_tile=M_tile, Nn_tile=Nn_tile, stable=stable
    ).cpu().numpy()
    return unif_is_drift_hats


# In[20]:


import gc, time
score_eval = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
score_eval_true_law = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_eval = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_eval_true_law = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_state_eval = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
score_state_eval = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}

score_eval_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
score_eval_true_law_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_eval_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_eval_true_law_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
nad_state_eval_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}
score_state_eval_std = {t: np.inf for t in ["8DLnz", "12DLnz", "20DLnz", "40DLnz"]}

for config in [lnz_40d_config, lnz_12d_config, lnz_20d_config, lnz_8d_config]:
    assert config.feat_thresh != 1.
    assert config.forcing_const == 1.25
    root_score_dir = root_dir
    label = "$\mu_{5}$"
    if "8DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/8DLnzChaos/"
        ts_type = "8DLnz"
    elif "12DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/12DLnzChaos/"
        ts_type = "12DLnz"
    elif "20DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/20DLnzChaos/"
        ts_type = "20DLnz"
    elif "40DLnz" in config.data_path:
        root_score_dir = root_dir + f"ExperimentResults/TSPM_Markovian/40DLnzChaos/"
        ts_type = "40DLnz"
    print(f"Starting {ts_type}\n")
    model_dir = "/".join(config.scoreNet_trained_path.split("/")[:-1]) + "/"
    entered = False
    for file in os.listdir(model_dir):
        if config.scoreNet_trained_path in os.path.join(model_dir, file) and ("EE" not in file and "Trk" not in file):
            good = ConditionalMarkovianTSPostMeanScoreMatching(
        *config.model_parameters)
            entered = True
            good.load_state_dict(torch.load(os.path.join(model_dir, file)))
    assert entered
    good = good.to(device_id)
    good.eval()

    # Prepare for Nadaraya
    is_obs, is_prevPath_obs, is_prevPath_incs = prepare_for_nadaraya(config=config, num_paths=num_paths)
    grid_1d = np.logspace(-3.55, -0.05, 30)
    xadd = np.logspace(-0.05, 1.0, 11)[1:]  # 10 values > -0.05
    xadd2 = np.logspace(1.0, 2.0, 11)[1:]  # 10 values > -0.05
    bws = np.concatenate([grid_1d, xadd, xadd2])
    bws = np.stack([bws for m in range(config.ndims)], axis=-1)
    bw = bws[-1, :]
    assert bw.shape[0] == config.ndims and len(bw.shape) == 1
    inv_H = np.diag(np.power(bw, -2))
    norm_const = 1 / np.sqrt((2. * np.pi) ** config.ndims * (1. / np.linalg.det(inv_H)))
    Nn_tile = 256000
    stable = True
    block_size = 2048

    all_true_paths, all_score_paths, all_nad_paths, num_time_steps = generate_synthetic_paths(config=config, device_id=device_id, good=good, M_tile=block_size, Nn_tile=Nn_tile, stable=stable, prevPath_observations=is_prevPath_obs, prevPath_incs=is_prevPath_incs, inv_H=inv_H, norm_const=norm_const)
    all_true_paths = all_true_paths.reshape((-1, num_time_steps+1, config.ts_dims), order="C")
    all_score_paths = all_score_paths.reshape((-1, num_time_steps+1, config.ts_dims), order="C")
    all_nad_paths = all_nad_paths.reshape((-1, num_time_steps+1, config.ts_dims), order="C")
    BB, TT, DD = all_score_paths.shape
    TT -= 1
    all_true_states = all_true_paths[:, 1:,:].reshape((-1, config.ts_dims), order="C")
    all_score_states = all_score_paths[:, 1:,:].reshape((-1, config.ts_dims), order="C")
    all_nad_states = all_nad_paths[:, 1:,:].reshape((-1, config.ts_dims), order="C")

    true_drift = true_drifts(state=all_true_states, device_id=device_id,config=config).cpu().numpy()[:,0,:]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(5)
    all_score_drift_ests = np.zeros_like(true_drift)
    all_nad_drift_ests = np.zeros_like(true_drift)
    all_score_drift_ests_true_law = np.zeros_like(true_drift)
    all_nad_drift_ests_true_law = np.zeros_like(true_drift)
    score_state_eval[ts_type] = np.sqrt(np.nanmean(np.sum(np.power(all_true_paths - all_score_paths, 2), axis=-1), axis=0))
    nad_state_eval[ts_type] = np.sqrt(np.nanmean(np.sum(np.power(all_true_paths - all_nad_paths, 2), axis=-1), axis=0))
    score_state_eval_std[ts_type] = np.nanstd(np.sum((all_true_paths - all_score_paths) ** 2, axis=-1), axis=0, ddof=1)
    nad_state_eval_std[ts_type] = np.nanstd(np.sum((all_true_paths - all_nad_paths) ** 2, axis=-1), axis=0, ddof=1)
    for k in tqdm(range(0, all_score_states.shape[0], block_size)):
        curr_states = torch.tensor(all_score_states[k:k+block_size, :], device=device_id, dtype=torch.float32)
        drift_ests = experiment_MLP_DDims_drifts(config=config, Xs=curr_states, good=good, onlyGauss=False)
        drift_ests= drift_ests[:, -1, :, :].reshape(drift_ests.shape[0],drift_ests.shape[2],drift_ests.shape[
                                                                                                -1] * 1).mean(axis=1)
        all_score_drift_ests[k:k+block_size,:] = drift_ests
        del curr_states
        curr_states = torch.tensor(all_nad_states[k:k+block_size, :], device=device_id, dtype=torch.float32)
        nad_drift_est = run_nadaraya_single_bw(config=config, is_path_observations=is_obs, states=curr_states, M_tile=block_size, inv_H=inv_H, norm_const=norm_const,stable=stable, Nn_tile=Nn_tile)
        all_nad_drift_ests[k:k+block_size,:] = nad_drift_est
        del curr_states
        # Now evaluate on true path law
        curr_states = torch.tensor(all_true_states[k:k+block_size, :], device=device_id, dtype=torch.float32)
        drift_ests = experiment_MLP_DDims_drifts(config=config, Xs=curr_states, good=good, onlyGauss=False)
        drift_ests = drift_ests[:, -1, :, :].reshape(drift_ests.shape[0], drift_ests.shape[2], drift_ests.shape[
            -1] * 1).mean(axis=1)
        all_score_drift_ests_true_law[k:k + block_size, :] = drift_ests
        # Now evaluate on true path law
        nad_drift_est = run_nadaraya_single_bw(config=config, is_path_observations=is_obs, states=curr_states, M_tile=block_size, inv_H=inv_H, norm_const=norm_const,stable=stable, Nn_tile=Nn_tile)
        all_nad_drift_ests_true_law[k:k+block_size,:] = nad_drift_est
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    mse = np.cumsum(np.nanmean(np.sum(np.power(
        true_drift.reshape(((BB, TT, DD)), order="C") - all_score_drift_ests.reshape(((BB, TT, DD)), order="C"), 2),
                                   axis=-1), axis=0)) / np.arange(1, TT + 1)
    score_eval[ts_type] = mse
    mse = np.cumsum(np.nanmean(np.sum(
        np.power(true_drift.reshape(((BB, TT, DD)), order="C") - all_nad_drift_ests.reshape(((BB, TT, DD)), order="C"),
                 2), axis=-1), axis=0)) / np.arange(1, TT + 1)
    nad_eval[ts_type] = mse
    mse = np.cumsum(np.nanmean(np.sum(np.power(
        true_drift.reshape(((BB, TT, DD)), order="C") - all_nad_drift_ests_true_law.reshape(((BB, TT, DD)), order="C"),
        2), axis=-1), axis=0)) / np.arange(1, TT + 1)
    nad_eval_true_law[ts_type] = mse
    mse = np.cumsum(np.nanmean(np.sum(np.power(
        true_drift.reshape(((BB, TT, DD)), order="C") - all_score_drift_ests_true_law.reshape(((BB, TT, DD)),
                                                                                              order="C"), 2), axis=-1),
                            axis=0)) / np.arange(1, TT + 1)
    score_eval_true_law[ts_type] = mse

    # STD
    std = np.nanstd(np.cumsum(np.where(~np.isnan(se := np.sum((true_drift.reshape((BB, TT, DD),
                                                                                  order="C") - all_score_drift_ests.reshape(
        (BB, TT, DD), order="C")) ** 2, axis=-1)), se, 0.0), axis=1) / np.maximum(1, np.cumsum(~np.isnan(se), axis=1)),
                    axis=0, ddof=1)
    score_eval_std[ts_type] = std
    std = np.nanstd(np.cumsum(np.where(~np.isnan(se := np.sum((true_drift.reshape((BB, TT, DD),
                                                                                  order="C") - all_nad_drift_ests.reshape(
        (BB, TT, DD), order="C")) ** 2, axis=-1)), se, 0.0), axis=1) / np.maximum(1, np.cumsum(~np.isnan(se), axis=1)),
                    axis=0, ddof=1)

    nad_eval_std[ts_type] = std

    std = np.nanstd(np.cumsum(np.where(~np.isnan(se := np.sum((true_drift.reshape((BB, TT, DD),
                                                                                  order="C") - all_nad_drift_ests_true_law.reshape(
        (BB, TT, DD), order="C")) ** 2, axis=-1)), se, 0.0), axis=1) / np.maximum(1, np.cumsum(~np.isnan(se), axis=1)),
                    axis=0, ddof=1)
    nad_eval_true_law_std[ts_type] = std

    std = np.nanstd(np.cumsum(np.where(~np.isnan(se := np.sum((true_drift.reshape((BB, TT, DD),
                                                                                  order="C") - all_score_drift_ests_true_law.reshape(
        (BB, TT, DD), order="C")) ** 2, axis=-1)), se, 0.0), axis=1) / np.maximum(1, np.cumsum(~np.isnan(se), axis=1)),
                    axis=0, ddof=1)
    score_eval_true_law_std[ts_type] = std

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


# In[27]:


import pandas as pd
save_path = (project_config.ROOT_DIR + f"experiments/results/DLnzChaos_NewLongerDriftEvalExp_MSEs_{num_paths}NPaths").replace(
            ".", "")
pd.DataFrame.from_dict(score_eval).to_parquet(save_path + "_score_MSE.parquet")
pd.DataFrame.from_dict(nad_eval).to_parquet(save_path + "_nad_MSE.parquet")
pd.DataFrame.from_dict(nad_eval_true_law).to_parquet(save_path + "_nad_true_law_MSE.parquet")
pd.DataFrame.from_dict(score_eval_true_law).to_parquet(save_path + "_score_true_law_MSE.parquet")
pd.DataFrame.from_dict(score_state_eval).to_parquet(save_path + "_score_state_MSE.parquet")
pd.DataFrame.from_dict(nad_state_eval).to_parquet(save_path + "_nad_state_MSE.parquet")
pd.DataFrame.from_dict(score_eval_std).to_parquet(save_path + "_score_MSE_STD.parquet")
pd.DataFrame.from_dict(nad_eval_std).to_parquet(save_path + "_nad_MSE_STD.parquet")
pd.DataFrame.from_dict(nad_eval_true_law_std).to_parquet(save_path + "_nad_true_law_MSE_STD.parquet")
pd.DataFrame.from_dict(score_eval_true_law_std).to_parquet(save_path + "_score_true_law_MSE_STD.parquet")
pd.DataFrame.from_dict(score_state_eval_std).to_parquet(save_path + "_score_state_MSE_STD.parquet")
pd.DataFrame.from_dict(nad_state_eval_std).to_parquet(save_path + "_nad_state_MSE_STD.parquet")
print("Score vs Nadaraya Alt Law", "\n", score_eval, "\n", nad_eval, "End\n")
print("Score vs Nadaraya True Law", "\n", score_eval_true_law, "\n", nad_eval_true_law, "End\n")
print("Score vs Nadaraya State Eval", "\n", score_state_eval, "\n", nad_state_eval, "End\n")





