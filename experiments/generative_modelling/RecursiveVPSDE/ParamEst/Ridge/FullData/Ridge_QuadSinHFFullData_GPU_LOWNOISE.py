import numpy as np
import pandas as pd
import scipy
import torch

from configs import project_config
from configs.RecursiveVPSDE.Markovian_fQuadSinHF.recursive_Markovian_PostMeanScore_fQuadSinHF2_LowFTh_T256_H05_tl_110data_StbleTgt_LOWNOISE import \
    get_config


def true_drifts(device_id, config, state):
    state = torch.tensor(state, device=device_id, dtype=torch.float32)
    drift = -2. * config.quad_coeff * state + config.sin_coeff * config.sin_space_scale * torch.sin(
        config.sin_space_scale * state)
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
        print(np.min(np.linalg.eigvalsh(BTB + l * I)), l)
        inv = torch.linalg.inv(BTB + l * I) @ BTZ
        val = torch.abs(inv.T @ inv - const_t).item()  # scalar float
        return val

    x0 = float(max(0., -torch.min(torch.linalg.eigvalsh(BTB)).item()) + 1e-12)

    # Use a method with bounds and a finite-diff step that isn't annihilated
    opt = scipy.optimize.minimize(
        obj, x0,
        method="L-BFGS-B",
        bounds=[(1e-12, None)],
        options={"eps": 1e-4, "maxiter": 200}
    )

    lhat = np.inf
    while not opt.success and not np.allclose(lhat, opt.x):
        lhat = opt.x
        opt = scipy.optimize.minimize(
            obj, opt.x,
            method="L-BFGS-B",
            bounds=[(1e-12, None)],
            options={"eps": 1e-4, "maxiter": 200}
        )

    lhat = float(np.atleast_1d(opt.x)[0])
    a = torch.atleast_2d(torch.linalg.inv(BTB + lhat * I) @ BTZ)
    a = torch.as_tensor(a, device=device_id, dtype=torch.float32)
    return a



config = get_config()
config.feat_thresh = 1./500.
num_paths = 1024 if config.feat_thresh == 1. else 10240
assert num_paths == 10240
assert config.diffusion == 0.1
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
if paths.shape[1] == config.ts_length + 1:
    paths = paths[:, 1:]
    np.save(config.data_path, paths)
paths = np.concatenate(
    [np.repeat(np.array(config.initState).reshape((1, 1)), paths.shape[0], axis=0),
     paths], axis=1)
assert paths.shape == (num_paths, config.ts_length + 1)
device_id = _get_device()

KNs = np.arange(1, 60, 1)
AN = -0.15
BN = -AN
numXs = 1024
Xs = torch.linspace(AN - 0.05, BN + 0.05, numXs+1).reshape(1, -1)
LN = np.log(num_paths)
M = 3 if "BiPot" in config.data_path else 2
true_drift = true_drifts(device_id=device_id,config=config, state=Xs[:, :-1]).squeeze().cpu().numpy().reshape((numXs, config.ndims))
true_drift[Xs[:, :-1].flatten() < AN, :] = np.nan
true_drift[Xs[:, :-1].flatten() > BN, :] = np.nan
mses = {}
for idxKN in range(0,KNs.shape[0]):
    KN = KNs[idxKN]
    B = spline_basis(paths=paths, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
    Z = np.power(deltaT,-1)*np.diff(paths, axis=1).reshape((paths.shape[0]*(paths.shape[1]-1),1))
    Z = torch.tensor(Z, dtype=torch.float32, device=device_id)
    assert (B.shape[0] == Z.shape[0] and len(B.shape)==len(Z.shape) == 2)
    coeffs = find_optimal_Ridge_estimator_coeffs(B=B, Z=Z, KN=KN, LN=LN, M=M, device_id=device_id)
    unif_B = spline_basis(paths=Xs, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
    ridge_drift = construct_Ridge_estimator(coeffs=coeffs, B=unif_B, LN=LN, device_id=device_id).cpu().numpy().flatten().reshape((numXs, config.ndims))
    ridge_drift[Xs[:, :-1].flatten() < AN, :] = np.nan
    ridge_drift[Xs[:, :-1].flatten() > BN, :] = np.nan
    mse = np.nanmean(np.sum(np.power(ridge_drift - true_drift, 2), axis=-1), axis=-1)
    mses[KN] = [mse]
    print(KN, mse)

save_path = (
        project_config.ROOT_DIR + f"experiments/results/Ridge_fQuadSinHF_DriftEvalExp_{num_paths}NPaths_{config.deltaT:.3e}dT_Diff{config.diffusion:.1f}").replace(
    ".", "")
mses = (pd.DataFrame(mses)).T
mses.columns = mses.columns.astype(str)
mses.to_parquet(save_path + "_MSEs.parquet", engine="fastparquet")
Kidx = np.argmin(mses.values.flatten())
KN = KNs[Kidx]
B = spline_basis(paths=paths, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
Z = np.power(deltaT,-1)*np.diff(paths, axis=1).reshape((paths.shape[0]*(paths.shape[1]-1),1))
Z = torch.tensor(Z, dtype=torch.float32, device=device_id)

assert (B.shape[0] == Z.shape[0] and len(B.shape)==len(Z.shape) == 2)
coeffs = find_optimal_Ridge_estimator_coeffs(B=B, Z=Z, KN=KN, LN=LN, M=M, device_id=device_id)
unif_B = spline_basis(paths=Xs, KN=KN, AN=AN, BN=BN, M=M, device_id=device_id)
ridge_drift = construct_Ridge_estimator(coeffs=coeffs, B=unif_B, LN=LN, device_id=device_id).cpu().numpy().flatten().reshape((numXs, config.ndims))
ridge_drift[Xs[:, :-1].flatten() < AN, :] = np.nan
ridge_drift[Xs[:, :-1].flatten() > BN, :] = np.nan
np.save(save_path + "_drift_est.npy",ridge_drift)
np.save(save_path + "_true_drift.npy",true_drift)





