from typing import Tuple

import numpy as np
import torch
import multiprocessing as mp
from functools import partial

from matplotlib import pyplot as plt
from tqdm import tqdm

from configs import project_config
from src.classes.ClassFractionalBrownianNoise import FractionalBrownianNoise
from utils.math_functions import compute_sig_size, ts_signature_pipeline, time_aug, invisibility_reset


def ts_comp(t:int, batch:torch.Tensor, sig_trunc:int, times:torch.Tensor)->Tuple[int, torch.Tensor]:
    """
    Helper function for parallelised signature computation
    :param t: Time
    :param batch: Data
    :param sig_trunc: Signature truncation
    :param times: Time indices
    :return: Tuple of (time_id, tensor)
    """
    print(t)
    N, d = batch.shape[0], batch.shape[-1]
    if t == 0:
        return (t,ts_signature_pipeline(
            data_batch=torch.hstack([torch.zeros(size=(N, 1, d)).to(batch.device), batch[:, [t], :]]),
            trunc=sig_trunc, times=times))
    else:
        return (t,ts_signature_pipeline(data_batch=batch[:, :t, :], trunc=sig_trunc,
                                     times=times[1:, :]))


def create_historical_vectors(batch: torch.Tensor, sig_trunc: int, sig_dim:int)->torch.Tensor:
    """
    Create feature vectors using path signatures
        :return: Feature vectors for each timestamp
    """

    # batch shape (N_batches, Time Series Length, Input Size)
    # The historical vector for each t in (N_batches, t, Input Size) is (N_batches, t-20:t, Input Size)
    # Create new tensor of size (N_batches, Time Series Length, Input Size, 20, Input Size) so that each dimension
    # of the time series has a corresponding past of size (20, 1)
    N, T, d = batch.shape
    # Now attempt on a rolling basis across time (we start at time 1/T since starting point is excluded)
    times = torch.atleast_2d((torch.arange(1, T + 1) / T)).T.to(batch.device)
    """
    nproc = 256
    with mp.Pool(processes=nproc) as pool:
        result = pool.map(
            partial(ts_comp, batch=batch, sig_trunc=sig_trunc,times=times), range(0, T)
        )
        pool.close()
    tsres = sorted(result, key=lambda x: x[0])
    full_feats = torch.stack([ten for _, ten in tsres]).permute(1,0,2)
    """
    full_feats = torch.zeros((N, T, compute_sig_size(dim=sig_dim, trunc=sig_trunc)))
    full_feats[:, 0, 0] = 1.
    for t in tqdm(range(T)):
        # We want to compute features for the path up to BUT NOT including time "t", for t=0 we have no path
        # Further, because fBM starts at 0, the first fBm value is also the first "increment"
        if t > 0:
            full_feats[:, t, :] = ts_signature_pipeline(data_batch=batch[:, :t, :], trunc=sig_trunc,
                                                        times=times)

    # assert(np.all([torch.all(torch.abs(full_feats[i,:,:]-els[i,:,:])<1e-12) for i in range(N)]))
    # Feature tensor is of size (Num_TimeSeries, TimeSeriesLength, FeatureDim)
    # Note first element of features are all the same so we exclude them
    return full_feats[:, :, 1:]

def compute_avg_feat(data, config, Nsamples):
    data = torch.Tensor(np.atleast_3d(data.cumsum(axis=1)[:Nsamples, :]))
    N, T = data.shape[:2]
    feats = create_historical_vectors(batch=data, sig_trunc=config.sig_trunc, sig_dim=config.sig_dim).numpy()
    assert (feats.shape == (
        N, config.ts_length, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1))
    avg_feats = np.mean(feats, axis=0)
    return avg_feats

if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    Nsamples = 100
    config.sig_trunc = 4
    sigfeatdim =  compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc)-1
    data = np.array([FractionalBrownianNoise(H=0.7).circulant_simulation(N_samples=config.ts_length) for _ in range(Nsamples)])
    avg_long_feats = compute_avg_feat(data=data, config=config, Nsamples=Nsamples)
    import signatory
    N, T, d = data.shape
    timeaug = time_aug(torch.Tensor(data), torch.linspace(0, 1., 256))
    transformed = invisibility_reset(timeaug, ts_dim=d)

    """
    brownian_data = np.array([FractionalBrownianNoise(H=0.5).circulant_simulation(N_samples=config.ts_length) for _ in range(Nsamples)])
    avg_brownian_feats = compute_avg_feat(data=brownian_data, config=config, Nsamples=Nsamples)

    rough_data = np.array([FractionalBrownianNoise(H=0.2).circulant_simulation(N_samples=config.ts_length) for _ in range(Nsamples)])
    avg_rough_feats = compute_avg_feat(data=rough_data, config=config, Nsamples=Nsamples)

    dimspace = np.arange(1,sigfeatdim+1)
    timespace = np.arange(0, config.ts_length)
    for t in range(253, 256):  # config.ts_length):
        plt.scatter(dimspace[1::2], avg_long_feats[t, 1::2], label="H07")
        plt.scatter(dimspace[1::2], avg_brownian_feats[t, 1::2], label="H05")
        plt.scatter(dimspace[1::2], avg_rough_feats[t, 1::2], label="H02")
        plt.title("Feature at time {}".format(t))
        plt.legend()
        plt.show()
        plt.close()
    for d in range(1, sigfeatdim, 2):
        plt.scatter(timespace, avg_long_feats[:, d], label="H07", s=1.)
        plt.scatter(timespace, avg_brownian_feats[:, d], label="H05", s=1.)
        plt.scatter(timespace, avg_rough_feats[:, d], label="H02", s=1.)
        plt.title("Feature {} timeseries".format(d+1))
        plt.legend()
        plt.show()
        plt.close()

    # np.save(config.feat_path, feats, allow_pickle=True)
    """

