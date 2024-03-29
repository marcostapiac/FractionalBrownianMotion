import pickle

import numpy as np
import pandas as pd
import torch

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
    for t in range(T):
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
    # Data parameters
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalTimeSeriesScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    end_epoch = max(config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(end_epoch)))
    except FileNotFoundError as e:
        print("Error {}; no valid trained model found; proceeding to training\n".format(e))
        training_size = 2 + 0 * int(
            min(config.tdata_mult * sum(p.numel() for p in scoreModel.parameters() if p.requires_grad), 1200000))
        try:
            data = np.load(config.data_path, allow_pickle=True)
            assert (data.shape[0] >= training_size)
        except (FileNotFoundError, pickle.UnpicklingError, AssertionError) as e:
            print("Error {}; generating synthetic data\n".format(e))
            data = generate_fBn(T=config.ts_length, isUnitInterval=config.isUnitInterval, S=training_size,
                                H=config.hurst)
            np.save(config.data_path, data)
        if config.isfBm:
            data = data.cumsum(axis=1)[:training_size, :]
        else:
            data = data[:training_size, :]
        data = torch.Tensor(np.atleast_3d(data))
        assert (data.shape == (training_size, config.ts_length, config.ts_dims))
        feats = create_historical_vectors(batch=data, sig_trunc=config.sig_trunc)
        assert (feats.shape == (
        training_size, config.ts_length, compute_sig_size(dim=config.sig_dim, trunc=config.sig_trunc) - 1))
        datafeats = torch.concat([data, feats], dim=-1).float()
        # For recursive version, data should be (Batch Size, Sequence Length, Dimensions of Time Series)
        init_experiment(config=config)
        train_and_save_recursive_diffusion_model(data=datafeats, config=config, diffusion=diffusion,
                                                 scoreModel=scoreModel,
                                                 trainClass=ConditionalSignatureDiffusionModelTrainer)
        cleanup_experiment()

    for train_epoch in config.max_epochs:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
        final_paths = recursive_signature_reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                                           data_shape=(config.dataSize, config.ts_length, 1),
                                                           config=config)
        df = pd.DataFrame(final_paths)
        df.index = pd.MultiIndex.from_product(
            [["Final Time Samples"], [i for i in range(config.dataSize)]])
        df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip")
