import numpy as np
import pandas as pd
import scipy
import torch
from tqdm import tqdm

from configs import project_config
from configs.RecursiveVPSDE.Markovian_fSinLog.recursive_Markovian_PostMeanScore_fSinLog_LowFTh_T256_H05_tl_110data_StbleTgt_FULLDATA import \
    get_config


def true_drifts(device_id, config, state):
    state = torch.tensor(state, device=device_id, dtype=torch.float32)
    drift = (-torch.sin(config.sin_space_scale * state) * torch.log(
        1 + config.log_space_scale * torch.abs(state)) / config.sin_space_scale)
    return drift[:, np.newaxis, :]


def _get_device(device_str: str  = None):
    if device_str is not None:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def spline_basis(paths, device_id, KN, AN, BN, M):
    paths = torch.as_tensor(paths, dtype=torch.float32, device=device_id)
    assert (paths.shape[0] >= 1 and len(paths.shape) == 2)
    assert (AN < BN and KN > 0 and M > 0)

    def construct_ith_knot(i, AN, BN, KN):
        if i < 0:
            return AN
        elif i > KN:
            return BN
        else:
            return AN + i * (BN - AN) / KN

    def bspline(i, l, u, x, KN, M):
        if l == 0 and -M <= i <= KN + M - 1:
            return ((u[i] <= x) & (x < u[i + 1])).float()
        elif 1 <= l <= M and -M <= i <= KN + M - l - 1:
            num1 = (x - u[i]) / (u[i + l] - u[i])
            num1[torch.isinf(num1)] = 0.
            num2 = ((-x + u[i + l + 1]) / (u[i + l + 1] - u[i + 1]))
            num2[torch.isinf(num2)] = 0.
            return num1 * bspline(i=i,     l=l - 1, u=u, x=x, KN=KN, M=M) + \
                   num2 * bspline(i=i + 1, l=l - 1, u=u, x=x, KN=KN, M=M)

    # Knots as CPU Python scalars (not moved to GPU)
    knots = {i: construct_ith_knot(i, AN, BN, KN) for i in range(-M, KN + M + 1)}

    # Preserve original path handling
    if paths.shape[1] > 1:
        paths = paths[:, :-1].flatten()
    else:
        paths = paths.flatten()

    # Build basis columns on GPU; no NumPy hop
    cols = [bspline(i=i, l=M, u=knots, x=paths, KN=KN, M=M) for i in range(-M, KN)]
    basis = torch.stack(cols, dim=1).to(device=device_id, dtype=torch.float32)

    assert (basis.shape == (paths.shape[0], KN + M)), \
        f"Basis is shape {basis.shape} but should be {(paths.shape[0], KN + M)}"
    #assert torch.all(basis >= 0.)
    return basis

def construct_Ridge_estimator(coeffs, B, LN, device_id):
    LN = torch.tensor(LN, dtype=torch.float32, device=device_id)
    drift  = B@coeffs
    drift[torch.abs(drift) > torch.sqrt(LN)] = torch.sqrt(LN)*torch.sign(drift[torch.abs(drift) > torch.sqrt(LN)])
    return drift




def find_optimal_Ridge_estimator_coeffs(B, Z, KN, LN, M, device_id):
    # Heavy ops on GPU in float64
    B = B.to(device_id, dtype=torch.float64)
    Z = Z.to(device_id, dtype=torch.float64)

    BTB = B.T @ B
    BTZ = B.T @ Z
    dtype = BTB.dtype
    device = BTB.device

    const = (KN + M) * LN
    const_t = torch.tensor(const, device=device, dtype=dtype)

    # Optional: release big inputs if memory is tight
    del B, Z

    # Same logic as before
    if torch.all(torch.linalg.eigvalsh(BTB) > 0.):
        print("Matrix BTB is invertible\n")
        a = torch.linalg.inv(BTB) @ BTZ
        if (a.T @ a).item() <= const_t.item():
            print("L2 norm of coefficients automatically satisfies projection constraint\n")
            return torch.as_tensor(a, device=device_id, dtype=torch.float32)

    I = torch.eye(KN + M, device=device_id, dtype=torch.float64)

    def obj(larr):
        # keep optimizer precision
        l = torch.as_tensor(larr, device=device_id, dtype=torch.float64)
        inv = torch.linalg.inv(BTB + l * I) @ BTZ
        val = torch.abs(inv.T @ inv - const_t).item()  # scalar float
        return val

    x0 = float(max(0., -torch.min(torch.linalg.eigvalsh(BTB)).item()) + 1e-12)

    # Use a method with bounds and a finite-diff step that isn't annihilated
    opt = scipy.optimize.minimize(
        obj, x0,
        method="L-BFGS-B",
        bounds=[(0.0, None)],
        options={"eps": 1e-4, "maxiter": 200}
    )

    lhat = np.inf
    while not opt.success and not np.allclose(lhat, opt.x):
        lhat = opt.x
        opt = scipy.optimize.minimize(
            obj, opt.x,
            method="L-BFGS-B",
            bounds=[(0.0, None)],
            options={"eps": 1e-4, "maxiter": 200}
        )

    lhat = float(np.atleast_1d(opt.x)[0])
    a = torch.atleast_2d(torch.linalg.inv(BTB + lhat * I) @ BTZ)
    a = torch.as_tensor(a, device=device_id, dtype=torch.float32)
    return a


def generate_synthetic_paths(config, device_id, ridge_coeffs, AN, BN):
    rmse_quantile_nums = 1
    num_paths = 100
    num_time_steps = int(1 * config.ts_length)
    deltaT = config.deltaT
    all_true_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_ridge_states = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))

    all_true_drifts = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))
    all_ridge_drifts = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))

    all_ridge_drifts_at_true = np.zeros(shape=(rmse_quantile_nums, num_paths, 1 + num_time_steps, config.ndims))

    for quant_idx in (range(rmse_quantile_nums)):
        initial_state = np.repeat(np.atleast_2d(config.initState)[np.newaxis, :], num_paths, axis=0)
        assert (initial_state.shape == (num_paths, 1, config.ndims))

        true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        ridge_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

        true_drift = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        ridge_drifts = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

        ridge_drifts_at_true = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

        # Initialise the "true paths"
        true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
        # Initialise the "global score-based drift paths"
        ridge_states[:, [0], :] = true_states[:, [0],
                                  :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)
        # Euler-Maruyama Scheme for Tracking Errors
        for i in tqdm(range(1, num_time_steps + 1)):
            eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT) * config.diffusion

            assert (eps.shape == (num_paths, 1, config.ndims))
            true_mean = true_drifts(state=true_states[:, i - 1, :], device_id=device_id,
                                    config=config).cpu().numpy()
            denom = 1.
            x = torch.as_tensor(ridge_states[:, i - 1, :], device=device_id, dtype=torch.float32).contiguous()
            ridge_basis = spline_basis(paths=x, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
            ridge_mean = construct_Ridge_estimator(coeffs=ridge_coeffs, B=ridge_basis, LN=LN,
                                                   device_id=device_id).cpu().numpy()[:, np.newaxis, :]
            del x
            x = torch.as_tensor(true_states[:, i - 1, :], device=device_id, dtype=torch.float32).contiguous()
            local_ridge_basis = spline_basis(paths=x, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
            local_ridge_mean = construct_Ridge_estimator(coeffs=ridge_coeffs, B=local_ridge_basis, LN=LN,
                                                         device_id=device_id).cpu().numpy()[:, np.newaxis, :]
            del x

            true_states[:, [i], :] = (true_states[:, [i - 1], :] + true_mean * deltaT + eps) / denom
            ridge_states[:, [i], :] = (ridge_states[:, [i - 1], :] + ridge_mean * deltaT + eps) / denom

            true_drift[:, [i], :] = true_mean

            ridge_drifts[:, [i], :] = ridge_mean

            ridge_drifts_at_true[:, [i], :] = local_ridge_mean

        all_true_states[quant_idx, :, :, :] = true_states
        all_ridge_states[quant_idx, :, :, :] = ridge_states

        all_true_drifts[quant_idx, :, :, :] = true_drift
        all_ridge_drifts[quant_idx, :, :, :] = ridge_drifts

        all_ridge_drifts_at_true[quant_idx, :, :, :] = ridge_drifts_at_true

    return all_true_states, all_ridge_states, \
           all_true_drifts, all_ridge_drifts, \
           all_ridge_drifts_at_true, num_time_steps



config = get_config()
config.feat_thresh = 1./500.
num_paths = 1024 if config.feat_thresh == 1. else 10240
assert num_paths == 10240
num_time_steps = config.ts_length
isUnitInterval = True
diff = config.diffusion
initial_state = config.initState
rvs = None
H = config.hurst
deltaT = config.deltaT
t0 = config.t0
t1 = deltaT * num_time_steps
paths = np.load(config.data_path, allow_pickle=True)[:num_paths, :]
paths = np.concatenate(
    [np.repeat(np.array(config.initState).reshape((1, 1)), paths.shape[0], axis=0),
     paths], axis=1)
assert paths.shape == (num_paths, config.ts_length + 1)
device_id = _get_device()

KNs = np.arange(1, 60, 1)
AN = -1.5
BN = -AN
numXs = 1024
LN = np.log(num_paths)
M = 3 if "BiPot" in config.data_path else 2
mses = {}
save_path = (
        project_config.ROOT_DIR + f"experiments/results/Ridge_fSinLog_DriftEvalExp_{num_paths}NPaths_{config.deltaT:.3e}dT").replace(
    ".", "")
for idxKN in range(0,KNs.shape[0]):
    KN = KNs[idxKN]
    B = spline_basis(paths=paths, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
    Z = np.power(deltaT,-1)*np.diff(paths, axis=1).reshape((paths.shape[0]*(paths.shape[1]-1),1))
    Z = torch.tensor(Z, dtype=torch.float32, device=device_id)
    assert (B.shape[0] == Z.shape[0] and len(B.shape)==len(Z.shape) == 2)
    ridge_coeffs = find_optimal_Ridge_estimator_coeffs(B=B, Z=Z, KN=KN, LN=LN, M=M, device_id=device_id)
    all_true_paths,  _, all_true_drifts,  _, \
     all_ridge_drift_ests_true_law, num_time_steps = generate_synthetic_paths(
        config=config, device_id=device_id, ridge_coeffs=ridge_coeffs, AN=AN, BN=BN)
    all_true_drifts = all_true_drifts.reshape((-1, num_time_steps + 1, config.ts_dims), order="C")
    all_true_paths = all_true_paths.reshape((-1, num_time_steps + 1, config.ts_dims), order="C")
    # True MSE
    BB, TT, DD = all_true_drifts.shape
    mse = np.cumsum(np.nanmean(np.sum(np.power(
        all_true_drifts.reshape((BB, TT, DD), order="C") - all_ridge_drift_ests_true_law.reshape((BB, TT, DD),
                                                                                                 order="C"), 2),
        axis=-1),
        axis=0)) / np.arange(1, TT + 1)
    mses[KN] = [mse[-1]]
    if mse[-1] < np.min(list(mses.values())):
        np.save(save_path + f"_{KN}_drift_est.npy", all_ridge_drift_ests_true_law)
        np.save(save_path + f"_{KN}_true_drift.npy", all_true_drifts)
        np.save(save_path + f"_{KN}_true_paths.npy", all_true_paths)
    print(KN, mse)

mses = (pd.DataFrame(mses)).T
mses.columns = mses.columns.astype(str)
mses.to_parquet(save_path + "_MSEs.parquet", engine="fastparquet")

