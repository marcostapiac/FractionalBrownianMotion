import math
from multiprocessing import shared_memory

import os
import torch
import numpy as np
from scipy.special import eval_laguerre
from torch.nn.utils.rnn import pad_sequence

from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion


def stochastic_burgers_drift(
    a: np.ndarray,
    num_paths,
    config,
) -> np.ndarray:
    """
    IMEX drift for Burgers in Fourier space (vectorized over leading batch dims).

      f_IMEX(a) = [ N(a) - ν k^2 ⊙ a ] / [ 1 + ν k^2 dt ]

    Inputs:
      a       : (..., M) float array containing ONLY the specified component (real or imag).
                Supports single state (M,), time-batches (T, M), or multi-path batches (I, T, M), etc.
      config  : object with attrs: num_fourier_modes (M), nu, deltaT, and optionally kappa, real (bool).
                If config.real is True, `a` is treated as the real part; otherwise as the imaginary part.

    Returns:
      (..., M) float array for the SAME component as input (real if config.real else imag).
    """
    M   = int(config.num_fourier_modes)
    nu  = float(config.nu)
    dt  = float(config.deltaT)
    kap = float(getattr(config, "kappa", 1.0))
    real_input = config.real

    # Validate and flatten batch
    a = np.asarray(a, dtype=np.float64)
    if a.shape[-1] != M:
        raise ValueError(f"Last dimension of a must be M={M}, got {a.shape[-1]}")
    batch_shape = a.shape[:-1]
    B = int(np.prod(batch_shape)) if batch_shape else 1
    a_flat = a.reshape(B, M)

    # Build complex spectrum with missing component = 0
    a_c = (a_flat.astype(np.complex128)
           if real_input
           else (1j * a_flat).astype(np.complex128))

    K = M - 1
    m = np.arange(M, dtype=np.float64)
    k2 = (kap * m) ** 2
    den = (1.0 + nu * k2 * dt).reshape(1, M)

    # Nonlinear term N(a) for each batch row
    Nh = np.zeros((B, M), dtype=np.complex128)
    for b in range(B):
        a_nonneg = a_c[b]
        # build symmetric spectrum a_full for indices -K..K
        a_full = np.empty(2 * K + 1, dtype=np.complex128)
        a_full[K] = a_nonneg[0]
        for mm in range(1, K + 1):
            a_full[K + mm] = a_nonneg[mm]
            a_full[K - mm] = np.conj(a_nonneg[mm])
        nh = np.zeros(M, dtype=np.complex128)
        for mm in range(1, K + 1):  # m=0 remains 0
            ik = 1j * kap * mm
            s = 0.0 + 0.0j
            for p in range(-K, K + 1):
                q_idx = mm - p
                if -K <= q_idx <= K:
                    s += a_full[K + p] * a_full[K + q_idx]
            nh[mm] = -0.5 * ik * s
        Nh[b] = nh

    out_c = (Nh - nu * k2.reshape(1, M) * a_c) / den
    out = out_c.real if real_input else out_c.imag
    return out.reshape(*batch_shape, M).astype(np.float64)

def build_q_nonneg(config):
    """
    Build per-mode variances q_m for m=0..K (nonnegative), with sum(q_m)=sigma_tot^2.
    Monotone decreasing in |k| for powerlaw/gaussian.
    """
    m = np.arange(config.num_fourier_modes, dtype=int)  # 0..K
    k_phys = m.astype(float)
    q = np.zeros_like(k_phys, dtype=float)
    if q.shape[0] > 1:
        q[1:] = 2 * config.nu * 5 * (np.abs(k_phys[1:]) ** (-config.alpha))
    return q
def process_IID_SBurgers_bandwidth(quant_idx, shape, inv_H, norm_const, true_drift, config, num_time_steps, num_state_paths,
                          deltaT, prevPath_name, path_incs_name, seed_seq):

    m = np.arange(config.num_fourier_modes, dtype=int)  # 0..K
    k_phys = m.astype(float)
    q = build_q_nonneg(config=config)
    rng = np.random.default_rng(seed_seq)  # This RNG can generate all needed arrays
    dtype =  np.float64#np.complex128

    # Attach to the shared memory blocks by name.
    shm_prev = shared_memory.SharedMemory(name=prevPath_name)
    shm_incs = shared_memory.SharedMemory(name=path_incs_name)

    # Create numpy array views from the shared memory (no copying happens here)
    prevPath_observations = np.ndarray(shape, dtype=dtype, buffer=shm_prev.buf)
    path_incs = np.ndarray(shape, dtype=dtype, buffer=shm_incs.buf)
    true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims),dtype=dtype)
    global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims),dtype=dtype)
    local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims),dtype=dtype)
    # Initialise the "true paths"
    true_states[:, [0], :] = config.initState.real if config.real else config.initState.imag
    global_states[:, [0], :] = config.initState.real if config.real else config.initState.imag
    local_states[:, [0], :] = config.initState.real if config.real else config.initState.imag
    for i in range(1, num_time_steps + 1):
        eps = np.zeros((num_state_paths, 1, config.num_fourier_modes), dtype=dtype)
        if config.real:
            # m=0 real noise
            eps[:, :, 0] = np.sqrt(q[0] * deltaT) * rng.standard_normal((num_state_paths, 1))
            if config.num_fourier_modes > 1:
                re = rng.standard_normal((num_state_paths, 1, config.num_fourier_modes - 1))
                # for m>=1: Re(eta_m) ~ N(0, q_m/2)
                eps[:, :, 1:] = np.sqrt((q[1:] * deltaT) / 2.0)[None, None, :] * re
        else:
            # imaginary component: no m=0 noise; only Im(eta_m)
            # eps[:, :, 0] stays 0
            if config.num_fourier_modes > 1:
                im = rng.standard_normal((num_state_paths, 1, config.num_fourier_modes - 1))
                # for m>=1: Im(eta_m) ~ N(0, q_m/2)
                eps[:, :, 1:] = np.sqrt((q[1:] * deltaT) / 2.0)[None, None, :] * im

        # Exponential Euler update:
        # a_{n+1} = E * a_n + B * N(a_n) + F * eta   (elementwise over modes)
        denom = 1.0 + config.nu * (k_phys ** 2) * deltaT

        true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)[:, np.newaxis, :]
        global_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H, norm_const=norm_const,
                                               x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=False)[:, np.newaxis, :]
        local_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H,
                                               norm_const=norm_const,
                                               x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=False)[:, np.newaxis, :]
        print(true_mean.shape)
        true_states[:, [i], :] = (true_states[:, [i - 1], :] + true_mean * deltaT + eps) / denom
        global_states[:, [i], :] = (global_states[:, [i - 1], :] + global_mean * deltaT + eps)/denom
        local_states[:, [i], :] = (true_states[:, [i - 1], :] + local_mean * deltaT + eps)/denom
    return {quant_idx: (true_states, global_states, local_states)}


def LSTM_1D_drifts(config, PM):
    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    PM = PM.to(device)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    ts_step = config.deltaT
    print(config.scoreNet_trained_path)
    Xshape = config.ts_length
    num_taus = 100

    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                     steps=Ndiff_discretisation).to(device)

    Xs = torch.linspace(-1.5, 1.5, steps=Xshape)
    features = find_LSTM_feature_vectors_oneDTS(Xs=Xs, score_model=PM, device=device, config=config)
    num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in Xs}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    final_vec_mu_hats = np.zeros(
        (Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims

    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)
    # ts = []
    es = 1
    # mu_hats_mean = np.zeros((tot_num_feats, num_taus))
    # mu_hats_std = np.zeros((tot_num_feats, num_taus))
    difftime_idx = num_diff_times - 1

    PM.eval()
    while difftime_idx >= num_diff_times - es:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)
        with torch.no_grad():
            PM_output = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * tot_num_feats, 1, config.ts_dims))
        A = final_mu_hats.reshape((num_taus, tot_num_feats, config.ts_dims))
        split_tensors = torch.split(A, list(num_feats_per_x.values()), dim=1)
        # Compute means along the column dimension
        means = torch.stack([t.mean(dim=1) for t in split_tensors], dim=1)
        assert (means.shape == (num_taus, Xshape, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        final_vec_mu_hats[:, difftime_idx, :] = means.permute((1, 0, 2)).cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
    return final_vec_mu_hats[:, -es:, :, :]


def experiment_MLP_DDims_drifts(config, Xs, good, onlyGauss=False):
    # device setup
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if getattr(config, "has_cuda",
                                                                                     False) else torch.device("cpu")
    good = good.to(device).eval()

    # constants
    ts_step = config.deltaT
    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    num_taus = 100  # as in your code
    es = 1  # as in your code
    dtype = torch.float32
    chunk_size = 1024
    # inputs
    Xs = torch.as_tensor(Xs, device=device, dtype=dtype)
    Xshape = Xs.shape[0]

    # diffusion scaffolding (unchanged)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time, steps=Ndiff_discretisation,
                                     device=device)

    final_vec_mu_hats = torch.empty((Xshape, es, num_taus, config.ts_dims), dtype=dtype, device="cpu")
    # If you ever set es>1, we must carry vec_Z_taus across diffusion steps per chunk.
    # We'll implement a generic mechanism that also works for es=1 with minimal overhead.
    n_chunks = math.ceil(Xshape / chunk_size)
    prev_states_cpu = [None] * n_chunks  # store per-chunk vec_Z_taus on CPU between steps

    insert_idx = -1
    with torch.inference_mode():
        for step_i, difftime_idx in enumerate(range(num_diff_times - 1, num_diff_times - es - 1, -1)):
            # scalar time for this step
            d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1]
            for chunk_id in range(n_chunks):
                start = chunk_id * chunk_size
                end = min(start + chunk_size, Xshape)
                n = end - start
                if n <= 0:
                    continue

                # features for this chunk: (n, 1, F)
                feats = Xs[start:end].view(n, 1, -1)

                # latent state for this chunk at current step
                if step_i == 0:  # first step => prior sample
                    vec_Z_taus = diffusion.prior_sampling(shape=(n * num_taus, 1, config.ts_dims)).to(device=device,
                                                                                                      dtype=dtype)
                else:  # subsequent steps => reload state
                    vec_Z_taus = prev_states_cpu[chunk_id].to(device)

                # times
                diff_times = d.expand(n)  # (n,)
                eff_times = diffusion.get_eff_times(diff_times=diff_times).view(n, 1, 1)  # (n,1,1)
                vec_diff_times = diff_times.repeat(num_taus)  # (n*num_taus,)
                vec_eff_times = eff_times.repeat(num_taus, 1, 1)  # (n*num_taus,1,1)
                vec_conditioner = feats.repeat(num_taus, 1, 1)  # (n*num_taus,1,F)

                # forward pass
                if onlyGauss:
                    scoreEval_vec_Z_taus = torch.randn_like(vec_Z_taus)
                else:
                    scoreEval_vec_Z_taus = vec_Z_taus

                vec_predicted_score = good.forward(
                    inputs=scoreEval_vec_Z_taus,
                    times=vec_diff_times,
                    conditioner=vec_conditioner,
                    eff_times=vec_eff_times
                )

                # scalar coefficients (avoid big expanded tensors)
                beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0])
                sigma2_taus = 1.0 - beta_taus.pow(2)
                if "PM" in config.scoreNet_trained_path:
                    predicted_score = -scoreEval_vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * vec_predicted_score
                else:
                    predicted_score = vec_predicted_score
                vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(
                    x=vec_Z_taus,
                    predicted_score=predicted_score,
                    diff_index=torch.tensor([int(num_diff_times - 1 - difftime_idx)], device=device),
                    max_diff_steps=Ndiff_discretisation
                )

                if "PM" in config.scoreNet_trained_path:
                    final_mu_hats = (-beta_taus * scoreEval_vec_Z_taus / sigma2_taus) + (
                            ((sigma2_taus + (beta_taus * config.diffusion).pow(2) * ts_step) / (
                                        ts_step * sigma2_taus)) * vec_predicted_score
                    )
                else:
                    final_mu_hats = (scoreEval_vec_Z_taus / (ts_step * beta_taus)) + (
                            ((sigma2_taus + (beta_taus * config.diffusion).pow(2) * ts_step) / (
                                        ts_step * beta_taus)) * vec_scores
                    )

                # reshape to (n, num_taus, ts_dims) then place into output
                means = final_mu_hats.view(num_taus, n, config.ts_dims).permute(1, 0, 2)  # (n, num_taus, ts_dims)
                final_vec_mu_hats[start:end, insert_idx, :, :] = means.cpu()
                # update latent state for next diffusion step (if es>1)
                vec_z = torch.randn_like(vec_drift)
                next_state = (vec_drift + vec_diffParam * vec_z).to("cpu")
                prev_states_cpu[chunk_id] = next_state

                # free per-chunk temps promptly
                del feats, vec_conditioner, vec_predicted_score, scoreEval_vec_Z_taus
                del vec_scores, vec_drift, vec_diffParam, final_mu_hats, means, vec_Z_taus, next_state

            insert_idx -= 1
    # match your original return type
    return final_vec_mu_hats.numpy()
def MLP_1D_drifts(config, PM):
    assert config.ndims == 1
    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    PM = PM.to(device)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    ts_step = config.deltaT
    print(config.scoreNet_trained_path)
    Xshape = config.ts_length
    num_taus = 100

    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                     steps=Ndiff_discretisation).to(device)

    if "BiPot" not in config.data_path:
        if config.diffusion == .1:
            Xs = torch.linspace(-.15, .15, steps=Xshape)
        elif config.diffusion == 10.:
            Xs = torch.linspace(-12, 12, steps=Xshape)
        else:
            Xs = torch.linspace(-1.5, 1.5, steps=Xshape)
    elif "BiPot" in config.data_path:
        if config.diffusion == 1:
            Xs = torch.linspace(-1.5, 1.5, steps=Xshape)
        elif config.diffusion == 0.1:
            Xs = torch.linspace(-.3, .3, steps=Xshape)
        elif config.diffusion == 10.:
            Xs = torch.linspace(-4., 4., steps=Xshape)

    features_tensor = torch.stack([Xs for _ in range(1)], dim=0).reshape(Xshape * 1, 1, -1).to(device)
    final_vec_mu_hats = np.zeros(
        (Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims
    vec_Z_taus = diffusion.prior_sampling(shape=(Xshape * num_taus, 1, config.ts_dims)).to(device)

    # ts = []
    es = 1
    ts = []
    # mu_hats_mean = np.zeros((tot_num_feats, num_taus))
    # mu_hats_std = np.zeros((tot_num_feats, num_taus))
    difftime_idx = num_diff_times - 1

    PM.eval()
    while difftime_idx >= num_diff_times - es:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(Xshape)]).reshape(Xshape * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * Xshape)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * Xshape, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * Xshape,
            1, -1)
        with torch.no_grad():
            PM_output = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * Xshape, 1, config.ts_dims))

        means = final_mu_hats.reshape((num_taus, Xshape, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        final_vec_mu_hats[:, difftime_idx, :] = means.permute((1, 0, 2)).cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
    return final_vec_mu_hats[:, -es:, :, :]


def LSTM_2D_drifts(PM, config):
    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    PM = PM.to(device)

    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    ts_step = config.deltaT

    num_taus = 100

    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                     steps=Ndiff_discretisation).to(device)

    numXs = 25
    minx = -1.
    maxx = -0.9
    Xs = np.linspace(minx, maxx, numXs)
    miny = 1.
    maxy = 1.1
    Ys = np.linspace(miny, maxy, numXs)
    Xs, Ys = np.meshgrid(Xs, Ys)
    Xs = np.column_stack([Xs.ravel(), Ys.ravel()])
    Xshape = Xs.shape[0]
    features = find_LSTM_feature_vectors_multiDTS(Xs=Xs, score_model=PM, device=device, config=config)
    num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in Xs}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    final_vec_mu_hats = np.zeros(
        (Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims

    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)
    # ts = []
    es = 1
    # mu_hats_mean = np.zeros((tot_num_feats, num_taus))
    # mu_hats_std = np.zeros((tot_num_feats, num_taus))
    difftime_idx = num_diff_times - 1
    PM.eval()
    while difftime_idx >= num_diff_times - es:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)

        with torch.no_grad():
            PM_output = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * tot_num_feats, 1, config.ts_dims))
        A = final_mu_hats.reshape((num_taus, tot_num_feats, config.ts_dims))
        split_tensors = torch.split(A, list(num_feats_per_x.values()), dim=1)
        # Compute means along the column dimension
        means = torch.stack([t.mean(dim=1) for t in split_tensors], dim=1)
        assert (means.shape == (num_taus, Xshape, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        final_vec_mu_hats[:, difftime_idx, :] = means.permute((1, 0, 2)).cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
    return final_vec_mu_hats[:, -es:, :, :]


def MLP_fBiPotDDims_drifts(config, PM):
    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    PM = PM.to(device)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    ts_step = config.deltaT
    print(config.scoreNet_trained_path)
    Xshape = config.ts_length
    num_taus = 100

    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                     steps=Ndiff_discretisation).to(device)
    if config.ndims == 12:
        Xs = torch.concat([torch.linspace(-5, 5, steps=Xshape).reshape(-1,1),torch.linspace(-4.7, 4.7, steps=Xshape).reshape(-1,1),\
                             torch.linspace(-4.4, 4.4, steps=Xshape).reshape(-1,1), torch.linspace(-4.2, 4.2, steps=Xshape).reshape(-1,1),\
                             torch.linspace(-4.05, 4.05, steps=Xshape).reshape(-1,1), torch.linspace(-3.9, 3.9, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-3.7, 3.7, steps=Xshape).reshape(-1,1), torch.linspace(-3.6, 3.6, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-3.55, 3.55, steps=Xshape).reshape(-1,1), torch.linspace(-3.48, 3.48, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-3.4, 3.4, steps=Xshape).reshape(-1,1), torch.linspace(-3.4, 3.4, steps=Xshape).reshape(-1,1)], dim=1)
    elif config.ndims == 8:
        Xs = torch.concat([torch.linspace(-4.9, 4.9, steps=Xshape).reshape(-1,1), torch.linspace(-4.4, 4.4, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-4.05, 4.05, steps=Xshape).reshape(-1,1), torch.linspace(-3.9, 3.9, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-3.7, 3.7, steps=Xshape).reshape(-1,1), torch.linspace(-3.6, 3.6, steps=Xshape).reshape(-1,1), \
                             torch.linspace(-3.5, 3.5, steps=Xshape).reshape(-1,1), torch.linspace(-3.4, 3.4, steps=Xshape).reshape(-1,1)], dim=1)
    if config.diffusion == 0.1:
        Xs /= 5.
    elif config.diffusion == 1.:
        Xs /= 3.
    features_tensor = torch.stack([Xs for _ in range(1)], dim=0).reshape(Xshape * 1, 1, -1).to(device)
    final_vec_mu_hats = np.zeros(
        (Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims
    vec_Z_taus = diffusion.prior_sampling(shape=(Xshape * num_taus, 1, config.ts_dims)).to(device)

    # ts = []
    es = 1
    ts = []
    # mu_hats_mean = np.zeros((tot_num_feats, num_taus))
    # mu_hats_std = np.zeros((tot_num_feats, num_taus))
    difftime_idx = num_diff_times - 1
    PM.eval()
    while difftime_idx >= num_diff_times - es:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(Xshape)]).reshape(Xshape * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * Xshape)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * Xshape, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * Xshape,
            1, -1)
        with torch.no_grad():
            PM_output = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                             eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus/sigma2_taus + (beta_taus/sigma2_taus)*PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(beta_taus * config.diffusion,
                                                                                              2) * ts_step)) / (
                                                                                    ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(beta_taus * config.diffusion,
                                                                                              2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * Xshape, 1, config.ts_dims))

        means = final_mu_hats.reshape((num_taus, Xshape, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        final_vec_mu_hats[:, difftime_idx, :] = means.permute((1, 0, 2)).cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
    return final_vec_mu_hats[:, -es:, :, :]


def multivar_score_based_LSTM_drift(score_model, num_diff_times, diffusion, num_paths, prev, ts_step, config,
                                    device):
    """ Computes drift using LSTM score network when features obtained from in-sample data """
    score_model = score_model.to(device)
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    if prev[0, :].shape[0] > 1:
        features = find_LSTM_feature_vectors_multiDTS(Xs=prev, score_model=score_model, device=device, config=config)
        num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in prev}
    else:
        features = find_LSTM_feature_vectors_oneDTS(Xs=prev, score_model=score_model, device=device, config=config)
        num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in prev}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)

    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)
        score_model.eval()
        with torch.no_grad():
            PM_output = score_model.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                   eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * tot_num_feats, 1, config.ts_dims))
        A = final_mu_hats.reshape((num_taus, tot_num_feats, config.ts_dims))
        split_tensors = torch.split(A, list(num_feats_per_x.values()), dim=1)
        # Compute means along the column dimension
        means = torch.stack([t.mean(dim=1) for t in split_tensors], dim=1)
        assert (means.shape == (num_taus, num_paths, config.ts_dims))

        # print(vec_Z_taus.shape, vec_scores.shape)
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy()


def multivar_score_based_MLP_drift(score_model, num_diff_times, diffusion, num_paths, prev, ts_step, config,
                                   device):
    """ Computes drift using MLP score network when features obtained from in-sample data """
    score_model = score_model.to(device)
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    features_tensor = torch.stack([torch.tensor(prev, dtype=torch.float32) for _ in range(1)], dim=0).reshape(
        num_paths * 1, 1, -1).to(device)
    assert (features_tensor.shape[0] == num_paths)
    vec_Z_taus = diffusion.prior_sampling(shape=(num_paths * num_taus, 1, config.ts_dims)).to(device)

    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.stack([d for _ in range(num_paths)]).reshape(num_paths * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * num_paths)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * num_paths, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * num_paths,
            1, -1)
        score_model.eval()
        with torch.no_grad():
            PM_output = score_model.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * num_paths, 1, config.ts_dims))
        means = final_mu_hats.reshape((num_taus, num_paths, config.ts_dims))
        assert (means.shape == (num_taus, num_paths, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy()


def multivar_score_based_LSTM_drift_OOS(score_model, time_idx, h, c, num_diff_times, diffusion, num_paths, prev,
                                        ts_step, config,
                                        device):
    """ Computes drift using LSTM score network when features obtained from LSTM directly """

    score_model = score_model.to(device)
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    score_model.eval()
    with torch.no_grad():
        if time_idx == 0:
            features, (h, c) = score_model.rnn(torch.tensor(prev[:, np.newaxis, :], dtype=torch.float32).to(device),
                                               None)
        else:
            features, (h, c) = score_model.rnn(torch.tensor(prev[:, np.newaxis, :], dtype=torch.float32).to(device),
                                               (h, c))
    assert (prev[0, :].shape[0] == config.ts_dims)
    if prev[0, :].shape[0] > 1:
        features = {tuple(prev[i, :].squeeze().tolist()): features[i] for i in range(prev.shape[0])}
        num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in prev}
    else:
        features = {prev[i, :].item(): features[i] for i in range(prev.shape[0])}
        num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in prev}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)

    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)
        score_model.eval()
        with torch.no_grad():
            PM_output = score_model.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * tot_num_feats, 1, config.ts_dims))
        A = final_mu_hats.reshape((num_taus, tot_num_feats, config.ts_dims))
        split_tensors = torch.split(A, list(num_feats_per_x.values()), dim=1)
        # Compute means along the column dimension
        means = torch.stack([t.mean(dim=1) for t in split_tensors], dim=1)
        assert (means.shape == (num_taus, num_paths, config.ts_dims))

        # print(vec_Z_taus.shape, vec_scores.shape)
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy(), h, c


def drifttrack_cummse(true, local, deltaT):
    all_true_states = true / np.sqrt(deltaT)
    all_local_states = local / np.sqrt(deltaT)
    all_local_errors = np.cumsum(np.mean(np.power(all_true_states - all_local_states, 2), axis=(1, 3)),
                                 axis=-1) / np.arange(1, all_local_states.shape[2] + 1)
    total_local_errors = np.mean(all_local_errors, axis=0)
    return total_local_errors[-1]

def drifttrack_mse(true, local, deltaT):
    return np.sqrt(np.mean(np.sum(np.power(true - local, 2), axis=-1), axis=(0, 1)))[-1]


def driftevalexp_mse_ignore_nans(true, pred):
    assert (true.shape[0] == pred.shape[0])
    assert len(true.shape) == len(pred.shape)
    assert len(true.shape) == 1 or len(true.shape) == 2
    if len(true.shape) == 2 and true.shape[-1] > 1:
        assert true.shape[1] == pred.shape[1]
        return np.nanmean(np.sum((true-pred)**2,axis=-1), axis=-1)
    else:
        true = true.flatten()
        pred = pred.flatten()
        mask = ~np.isnan(true) & ~np.isnan(pred)  # Ignore NaNs in both arrays
        return np.mean((true[mask] - pred[mask]) ** 2)


def multivar_score_based_MLP_drift_OOS(score_model, num_diff_times, diffusion, num_paths, prev,
                                       ts_step, config,
                                       device):
    """ Computes drift using MLP score network when features obtained from LSTM directly """

    score_model = score_model.to(device)
    num_taus = 1000
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    assert (prev[0, :].shape[0] == config.ts_dims)
    features_tensor = torch.stack([torch.tensor(prev, dtype=torch.float32) for _ in range(1)], dim=0).reshape(
        num_paths * 1, 1, -1).to(device)
    assert (features_tensor.shape[0] == num_paths)
    vec_Z_taus = torch.zeros(size=(num_paths * num_taus, 1, config.ts_dims)).to(device)#diffusion.prior_sampling(shape=(num_paths * num_taus, 1, config.ts_dims)).to(device)#
    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.stack([d for _ in range(num_paths)]).reshape(num_paths * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * num_paths)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * num_paths, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * num_paths,
            1, -1)
        score_model.eval()
        with torch.no_grad():
            PM_output = score_model.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                            eff_times=vec_eff_times)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma2_taus = (1. - torch.pow(beta_taus, 2)).to(device)
        sigma_taus = torch.pow(sigma2_taus, 0.5).to(device)

        if "PM" in config.scoreNet_trained_path:
            vec_predicted_score = -vec_Z_taus / sigma2_taus + (beta_taus / sigma2_taus) * PM_output
        else:
            vec_predicted_score = PM_output
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)

        if "PM" in config.scoreNet_trained_path:
            final_mu_hats = (-beta_taus*vec_Z_taus / (sigma2_taus)) + ((
                                                                              (torch.pow(sigma_taus, 2) + (
                                                                                      torch.pow(
                                                                                          beta_taus * config.diffusion,
                                                                                          2) * ts_step)) / (
                                                                                      ts_step * sigma2_taus)) * PM_output)
        else:
            final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                            (torch.pow(sigma_taus, 2) + (
                                                                                    torch.pow(
                                                                                        beta_taus * config.diffusion,
                                                                                        2) * ts_step)) / (
                                                                                    ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * num_paths, 1, config.ts_dims))
        means = final_mu_hats.reshape((num_taus, num_paths, config.ts_dims))
        assert (means.shape == (num_taus, num_paths, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy()


def find_LSTM_feature_vectors_multiDTS(Xs, score_model, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float).to(device)

    def process_single_threshold(x, score_model, device, dX_global):
        diff = sim_data_tensor - x.reshape(1, -1)  # shape: (M, N, D)
        diff = diff.norm(dim=-1)  # Result: (M, N)
        mask = diff <= torch.min(diff)

        """mask = diff <=np.arccos(dX_global)
        thresh = dX_global
        while torch.sum(mask) == 0:
            thresh = np.cos(np.arccos(thresh)*2)
            mask = diff <= np.arccos(thresh)
        assert (torch.sum(mask) > 0)"""

        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)
        sequences = []
        js = []
        for k in range(min(100, indices.shape[0])):
            idx = indices[k, :]
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1, :]
            sequences.append(seq)
            js.append(len(seq))
        outputs = []
        score_model = score_model.to(device)
        score_model.eval()
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan).to(device)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            # padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = score_model.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    features_Xs = {}
    dX_global = np.cos(1. / 5000)
    for i in range(Xs.shape[0]):
        x = Xs[i, :].reshape(-1, 1)
        x_val, out = process_single_threshold(torch.tensor(x).to(device, dtype=torch.float), device=device,
                                              score_model=score_model, dX_global=dX_global)
        assert (len(out) > 0)
        features_Xs[tuple(x.squeeze().tolist())] = out
    return features_Xs


def find_LSTM_feature_vectors_oneDTS(Xs, score_model, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float).to(device)

    def process_single_threshold(x, score_model, device, dX):
        # Compute the mask over the entire sim_data matrix
        diff = sim_data_tensor - x
        mask = torch.abs(diff) <= torch.min(torch.abs(diff))
        """
        xmin = x - dX
        xmax = x + dX
        mask = (sim_data_tensor >= xmin) & (sim_data_tensor <= xmax)
        while torch.sum(mask) == 0:
            dX *= 2
            xmin = x - dX
            xmax = x + dX
            # Compute the mask over the entire sim_data matrix
            mask = (sim_data_tensor >= xmin) & (sim_data_tensor <= xmax)
        """
        assert torch.sum(mask) > 0
        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)

        sequences = []
        js = []
        for k in range(min(100, indices.shape[0])):
            idx = indices[k, :]
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1]
            sequences.append(seq)
            js.append(len(seq))

        outputs = []
        score_model = score_model.to(device)
        score_model.eval()
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = score_model.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    # Option 1: Process sequentially (using tqdm)
    features_Xs = {}
    assert (len(Xs.shape) == 1 or Xs.shape[-1] == 1)
    if len(Xs.shape) == 1:  # Domain RMSE
        dX_global = np.diff(Xs)[0] / 5000
        assert (((Xs[1] - Xs[0]) / 5000) == dX_global)
        for x in (Xs):
            x_val, out = process_single_threshold(x, device=device, score_model=score_model, dX=dX_global)
            assert (len(out) > 0)
            features_Xs[x_val.item()] = out
    else:
        for i in range(Xs.shape[0]):
            dX_global = 1. / 5000
            x = Xs[i, :].reshape(-1, 1)
            assert (x.shape[-1] == 1 and x.shape[0] == 1)
            x_val, out = process_single_threshold(x[0, 0], score_model=score_model, device=device, dX=dX_global)
            assert (len(out) > 0)
            features_Xs[x_val.item()] = out
    return features_Xs


def multivar_gaussian_kernel(inv_H, norm_const, x):
    """exponent = -0.5 * np.einsum('...i,ij,...j', x, inv_H, x)
    print(exponent.shape, x.shape)
    return norm_const * np.exp(exponent)"""
    if torch.cuda.is_available():
        device = 0
    else:
        device = torch.device("cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)
    inv_H = torch.tensor(inv_H, dtype=torch.float32, device=device)
    y = torch.matmul(x, inv_H)  # shape: (N, T1, T2, D)
    # Compute the dot product along the last dimension.
    # This is equivalent to the einsum: '...i, ...i'
    exponent = -0.5 * torch.sum(x * y, dim=-1)  # shape: (N, T1, T2)
    # Return the computed Gaussian kernel.
    res = (norm_const * torch.exp(exponent)).cpu().numpy()
    exponent = exponent.to("cpu")
    x = x.to("cpu")
    inv_H = inv_H.to("cpu")
    y = y.to("cpu")
    return res


def IID_NW_multivar_estimator(prevPath_observations, path_incs, inv_H, norm_const, x, t1, t0, truncate):
    if len(prevPath_observations.shape) > 2:
        N, n, d = prevPath_observations.shape
        kernel_weights_unnorm = multivar_gaussian_kernel(inv_H=inv_H, norm_const=norm_const,
                                                         x=prevPath_observations[:, :, np.newaxis, :] - x[np.newaxis,
                                                                                                        np.newaxis,
                                                                                                        :, :])
        denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
        assert (denominator.shape == (x.shape[0], 1))
        numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, :], axis=(1, 0)) / N * (
                t1 - t0)
    elif len(prevPath_observations.shape) == 2:
        N, n = prevPath_observations.shape
        kernel_weights_unnorm = multivar_gaussian_kernel(inv_H=inv_H, norm_const=norm_const,
                                                         x=prevPath_observations[:, :, np.newaxis, np.newaxis] - x[
                                                                                                                 np.newaxis,
                                                                                                                 np.newaxis,
                                                                                                                 :, :])
        denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
        assert (denominator.shape == (x.shape[0], 1))
        numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, np.newaxis],
                           axis=(1, 0)) / N * (
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

def process_IID_bandwidth(quant_idx, shape, inv_H, norm_const, true_drift, config, num_time_steps, num_state_paths,
                          deltaT, prevPath_name, path_incs_name, seed_seq):
    rng = np.random.default_rng(seed_seq)  # This RNG can generate all needed arrays
    dtype = np.complex128 if "Burgers" in config.data_path else np.float64

    # Attach to the shared memory blocks by name.
    shm_prev = shared_memory.SharedMemory(name=prevPath_name)
    shm_incs = shared_memory.SharedMemory(name=path_incs_name)

    # Create numpy array views from the shared memory (no copying happens here)
    prevPath_observations = np.ndarray(shape, dtype=dtype, buffer=shm_prev.buf)
    path_incs = np.ndarray(shape, dtype=dtype, buffer=shm_incs.buf)
    true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    # Initialise the "true paths"
    true_states[:, [0], :] = config.initState
    global_states[:, [0], :] = config.initState
    local_states[:, [0], :] = config.initState
    for i in range(1, num_time_steps + 1):
        eps = rng.normal(loc=0.0, scale=1.0, size=(num_state_paths, 1, config.ndims))
        eps *= np.sqrt(deltaT) * config.diffusion
        assert (eps.shape == (num_state_paths, 1, config.ndims))
        true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)
        global_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H, norm_const=norm_const,
                                               x=global_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=False)[:, np.newaxis, :]
        local_mean = IID_NW_multivar_estimator(prevPath_observations=prevPath_observations, inv_H=inv_H,
                                               norm_const=norm_const,
                                               x=true_states[:, i - 1, :], path_incs=path_incs, t1=config.t1,
                                               t0=config.t0, truncate=False)[:, np.newaxis, :]
        print(true_mean.shape)
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
    return {quant_idx: (true_states, global_states, local_states)}


def hermite_basis(R, paths):
    assert (paths.shape[0] >= 1 and len(paths.shape) == 2)
    basis = np.zeros((paths.shape[0], paths.shape[1], R))
    polynomials = np.zeros((paths.shape[0], paths.shape[1], R))
    for i in range(R):
        if i == 0:
            polynomials[:, :, i] = np.ones_like(paths)
        elif i == 1:
            polynomials[:, :, i] = paths
        else:
            polynomials[:, :, i] = 2. * paths * polynomials[:, :, i - 1] - 2. * (i - 1) * polynomials[:, :, i - 2]
        basis[:, :, i] = np.power((np.power(2, i) * np.sqrt(np.pi) * math.factorial(i)), -0.5) * polynomials[:, :,
                                                                                                 i] * np.exp(
            -np.power(paths, 2) / 2.)
    return basis


def laguerre_basis(R, paths):
    basis = np.zeros((paths.shape[0], paths.shape[1], R))
    for i in range(R):
        basis[:, :, i] = np.sqrt(2.) * eval_laguerre(i, 2. * paths) * np.exp(-paths) * (paths >= 0.)
    return basis


def construct_Z_vector(R, T, basis, paths):
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N = basis.shape[0]
    dXs = np.diff(paths, axis=1) / T
    Z = np.diagonal(basis.transpose((2, 0, 1)) @ (dXs.T), axis1=1, axis2=2)
    assert (Z.shape == (R, N))
    Z = Z.mean(axis=-1, keepdims=True)
    assert (Z.shape == (R, 1)), f"Z vector is shape {Z.shape} but should be {(R, 1)}"
    return Z


def construct_Phi_matrix(R, deltaT, T, basis, paths):
    assert (basis.shape[0] == paths.shape[0])
    assert (basis.shape[1] == paths.shape[1])
    basis = basis[:, :-1, :]
    assert (basis.shape[-1] == R)
    N, _ = basis.shape[:2]
    deltaT /= T
    intermediate = deltaT * basis.transpose((0, 2, 1)) @ basis
    assert intermediate.shape == (
        N, R, R), f"Intermidate matrix is shape {intermediate.shape} but shoould be {(N, R, R)}"
    for i in range(N):
        es = np.linalg.eigvalsh(intermediate[i, :, :]) >= 0.
        assert (np.all(es)), f"Submat at {i} is not PD, for R={R}"
    Phi = deltaT * (basis.transpose((0, 2, 1)) @ basis)
    assert (Phi.shape == (N, R, R))
    Phi = Phi.mean(axis=0, keepdims=False)
    assert (Phi.shape == (R, R)), f"Phi matrix is shape {Phi.shape} but should be {(R, R)}"
    assert np.all(np.linalg.eigvalsh(Phi) >= 0.), f"Phi matrix is not PD"
    return Phi


def estimate_coefficients(R, deltaT, t1, basis, paths, Phi=None):
    Z = construct_Z_vector(R=R, T=t1, basis=basis, paths=paths)
    if Phi is None:
        Phi = construct_Phi_matrix(R=R, deltaT=deltaT, T=t1, basis=basis, paths=paths)
    theta_hat = np.linalg.solve(Phi, Z)
    assert (theta_hat.shape == (R, 1))
    return theta_hat


def construct_Hermite_drift(basis, coefficients):
    b_hat = (basis @ coefficients).squeeze(-1)
    assert (b_hat.shape == basis.shape[:2]), f"b_hat should be shape {basis.shape[:2]}, but has shape {b_hat.shape}"
    return b_hat


def basis_number_selection(paths, num_paths, num_time_steps, deltaT, t1):
    poss_Rs = np.arange(1, 19)
    kappa = 1.  # See just above Section 5
    cvs = []
    for r in poss_Rs:
        print(cvs, r)
        basis = hermite_basis(R=r, paths=paths)
        try:
            Phi = construct_Phi_matrix(R=r, deltaT=deltaT, T=t1, basis=basis, paths=paths)
        except AssertionError:
            cvs.append(np.inf)
            continue
        coeffs = estimate_coefficients(R=r, deltaT=deltaT, basis=basis, paths=paths, t1=t1, Phi=Phi)
        bhat = np.power(construct_Hermite_drift(basis=basis, coefficients=coeffs), 2)
        bhat_norm = np.mean(np.sum(bhat * deltaT / t1, axis=-1))
        inv_Phi = np.linalg.inv(Phi)
        s = np.sqrt(np.max(np.linalg.eigvalsh(inv_Phi @ inv_Phi.T)))
        if np.power(s, 0.25) * r > num_paths * t1:
            cvs.append(np.inf)
        else:
            # Note that since we force \sigma = 1., then the m,sigma^2 matrix is all ones
            PPt = inv_Phi @ np.ones_like(inv_Phi)
            s_p = np.sqrt(np.max(np.linalg.eigvalsh(PPt @ PPt.T)))
            pen = kappa * s_p / (num_paths * num_time_steps * deltaT)
            cvs.append(-bhat_norm + pen)

    # R = basis_number_selection(paths=paths, num_paths=num_paths, num_time_steps=num_time_steps, deltaT=deltaT, t1=t1)
    # print(R)
    return poss_Rs[np.argmin(cvs)]


def process_single_R_hermite(quant_idx, R, shape, true_drift, config, num_time_steps, num_state_paths, deltaT,
                             path_name, seed_seq):
    rng = np.random.default_rng(seed_seq)  # This RNG can generate all needed arrays

    # Attach to the shared memory blocks by name.
    shm_path = shared_memory.SharedMemory(name=path_name)

    # Create numpy array views from the shared memory (no copying happens here)
    paths = np.ndarray(shape, dtype=np.float64, buffer=shm_path.buf)

    basis = hermite_basis(R=R, paths=paths)
    coeffs = estimate_coefficients(R=R, deltaT=deltaT, basis=basis, paths=paths, t1=config.t1, Phi=None)
    true_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    global_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    local_states = np.zeros(shape=(num_state_paths, 1 + num_time_steps, config.ndims))
    # Initialise the "true paths"
    true_states[:, [0], :] = config.initState
    global_states[:, [0], :] = config.initState
    local_states[:, [0], :] = config.initState

    for i in range(1, num_time_steps + 1):
        eps = rng.normal(loc=0.0, scale=1.0, size=(num_state_paths, 1, config.ndims))
        eps *= np.sqrt(deltaT) * config.diffusion
        assert (eps.shape == (num_state_paths, 1, config.ndims))
        true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_state_paths, config=config)
        global_basis = hermite_basis(R=R, paths=global_states[:, i - 1, :])
        global_mean = construct_Hermite_drift(basis=global_basis, coefficients=coeffs)[:, np.newaxis, :]
        local_basis = hermite_basis(R=R, paths=true_states[:, i - 1, :])
        local_mean = construct_Hermite_drift(basis=local_basis, coefficients=coeffs)[:, np.newaxis, :]
        true_states[:, [i], :] = true_states[:, [i - 1], :] + true_mean * deltaT + eps
        global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
        local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps
    return {quant_idx: (true_states, global_states, local_states)}
