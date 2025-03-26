import numpy as np
from configs import project_config
from tqdm import tqdm
from scipy.stats import norm
from src.classes.ClassFractionalMullerBrown import FractionalMullerBrown
from configs.RecursiveVPSDE.LSTM_fMullerBrown.recursive_LSTM_PostMeanScore_MullerBrown_T256_H05_tl_110data import \
    get_config


def gaussian_kernel(bw, x):
    return norm.pdf(x / bw) / bw


def multivar_gaussian_kernel(bw, x):
    D = x.shape[-1]
    inv_H = np.diag(np.power(bw, -2))
    norm_const = 1 / np.sqrt((2. * np.pi) ** D * (1. / np.linalg.det(inv_H)))
    exponent = -0.5 * np.einsum('...i,ij,...j', x, inv_H, x)
    return norm_const * np.exp(exponent)

def rmse_ignore_nans(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)  # Ignore NaNs in both arrays
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))


config = get_config()
print(config.deltaT)
num_paths = 10952
num_time_steps = config.ts_length
isUnitInterval = True
diff = config.diffusion
rvs = None
H = config.hurst
deltaT = config.deltaT
t0 = config.t0
t1 = deltaT * num_time_steps
initial_state = np.array(config.initState)
print(deltaT, t0, t1)


fMB = FractionalMullerBrown(initialState=np.array(initial_state), X0s=np.array(config.X0s), Y0s=np.array(config.Y0s),
                            diff=config.diffusion, Aks=np.array(config.Aks),
                            aks=np.array(config.aks), bks=np.array(config.bks), cks=np.array(config.cks))
is_path_observations = np.array(
    [fMB.euler_simulation(H=H, N=config.ts_length, deltaT=deltaT, X0=initial_state, Ms=None, gaussRvs=rvs,
                          t0=t0, t1=t1) for _ in (range(num_paths * 10))]).reshape(
    (num_paths * 10, config.ts_length + 1, config.ndims))

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


def IID_NW_multivar_estimator(prevPath_observations, path_incs, bw, x, t1, t0, truncate):
    N, n, d = prevPath_observations.shape
    kernel_weights_unnorm = multivar_gaussian_kernel(bw=bw, x=prevPath_observations[:, :, np.newaxis, :] - x[np.newaxis,
                                                                                                           np.newaxis,
                                                                                                           :, :])
    denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
    assert (denominator.shape == (x.shape[0], 1))
    numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, :], axis=(1, 0)) / N * (
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



assert (prevPath_observations.shape[1] * deltaT == (t1 - t0))

# Note that because b(x) = sin(x) is bounded, we take \epsilon = 0 hence we have following h_max
eps = 0.
log_h_min = np.log10(np.power(float(config.ts_length - 1), -(1. / (2. - eps))))


grid_1d = np.logspace(-4, -0.05, 40)
bws = np.stack([grid_1d for m in range(config.ndims)], axis=-1)
print(bws.shape)

numXs = 25
minx = -1.
maxx = -0.9
Xs = np.linspace(minx, maxx, numXs)
miny = 1.
maxy = 1.1
Ys = np.linspace(miny, maxy, numXs)
Xs, Ys = np.meshgrid(Xs, Ys)
Xs = np.column_stack([Xs.ravel(), Ys.ravel()])

num_dhats = 10
for bw in bws:
    unif_is_drift_hats = np.zeros((Xs.shape[0], num_dhats, config.ndims))

    for k in tqdm(range(num_dhats)):
        is_ss_path_observations = is_path_observations[np.random.choice(is_idxs, size=num_paths, replace=False), :]
        # Remember t0 = deltaT so X_t1 = is_ss_path_observations[:, 2] not is_ss_path_observations[:, 1]
        is_prevPath_observations = is_ss_path_observations[:, 1:-1]
        is_path_incs = np.diff(is_ss_path_observations, axis=1)[:, 1:]
        unif_is_drift_hats[:, k, :] = IID_NW_multivar_estimator(prevPath_observations=is_prevPath_observations, bw=bw, x=Xs,
                                                    path_incs=is_path_incs, t1=t1, t0=t0, truncate=True)
    save_path = (
            project_config.ROOT_DIR + f"experiments/results/IIDNadaraya_fMullerBrown_DriftEvalExp_{bw[0]}bw_{num_paths}NPaths_{config.t0}t0_{config.deltaT:.3e}dT").replace(
        ".", "")
    print(f"Save Path {save_path}\n")
    np.save(save_path + "_isdriftHats.npy", unif_is_drift_hats)
