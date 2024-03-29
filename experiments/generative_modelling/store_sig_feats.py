import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs import project_config
from src.classes.ClassConditionalSignatureDiffTrainer import ConditionalSignatureDiffusionModelTrainer
from src.generative_modelling.data_processing import train_and_save_recursive_diffusion_model
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTimeSeriesScoreMatching import \
    ConditionalTimeSeriesScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from utils.data_processing import cleanup_experiment, init_experiment
from utils.math_functions import generate_fBn, compute_sig_size, ts_signature_pipeline


def create_historical_vectors(batch: torch.Tensor, sig_trunc: int):
    """
    Create feature vectors using path signatures
        :return: Feature vectors for each timestamp
    """

    # batch shape (N_batches, Time Series Length, Input Size)
    # The historical vector for each t in (N_batches, t, Input Size) is (N_batches, t-20:t, Input Size)
    # Create new tensor of size (N_batches, Time Series Length, Input Size, 20, Input Size) so that each dimension
    # of the time series has a corresponding past of size (20, 1)
    N, T, d = batch.shape
    # Now attempt on a rolling basis across time
    times = (torch.atleast_2d((torch.arange(0, T + 1) / T)).T).to(batch.device)
    full_feats = torch.zeros(size=(N, T, compute_sig_size(dim=d + 1, trunc=sig_trunc))).to(batch.device)
    for t in tqdm(range(T)):
        if t == 0:
            full_feats[:, t, :] = ts_signature_pipeline(
                data_batch=torch.hstack([torch.zeros(size=(N, 1, d)).to(batch.device), batch[:, [t], :]]),
                trunc=sig_trunc, times=times)
        else:
            full_feats[:, t, :] = ts_signature_pipeline(data_batch=batch[:, :t, :], trunc=sig_trunc,
                                                        times=times[1:, :])

    # Feature tensor is of size (Num_TimeSeries, TimeSeriesLength, FeatureDim)
    # Note first element of features are all the same
    return full_feats[:, :, 1:]


if __name__ == "__main__":
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    data = np.load(config.data_path, allow_pickle=True)
    data = torch.Tensor(np.atleast_3d(data.cumsum(axis=1)))
    N, T = data.shape[:2]
    feats = create_historical_vectors(batch=data, sig_trunc=config.sig_trunc).numpy()
    assert (feats.shape == (
        N, config.ts_length, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1))
    np.save(project_config.ROOT_DIR + "data/fBm_T{}_SigTrunc{}_SigDim{}.npy".format(T, config.sig_trunc,
                                                                                           config.sig_dim), feats,
            allow_pickle=True)
