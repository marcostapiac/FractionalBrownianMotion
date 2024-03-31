from typing import Tuple

import numpy as np
import torch
import multiprocessing as mp
from functools import partial

from tqdm import tqdm

from configs import project_config
from utils.math_functions import compute_sig_size, ts_signature_pipeline


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


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    data = np.load(config.data_path, allow_pickle=True)
    data = torch.Tensor(np.atleast_3d(data.cumsum(axis=1)[:200000,:]))
    N, T = data.shape[:2]
    feats = create_historical_vectors(batch=data, sig_trunc=config.sig_trunc, sig_dim=config.sig_dim).numpy()
    assert (feats.shape == (
        N, config.ts_length, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1))
    np.save(config.feat_path, feats, allow_pickle=True)
