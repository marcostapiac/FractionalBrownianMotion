import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalTSScoreMatching import \
    ConditionalTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP


def exact_fBm_features():
    from configs.RecursiveVPSDE.recursive_fBm_T256_H07_tl_5data import get_config
    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    config.dataSize = 40000
    scoreModel = ConditionalTSScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    # init_experiment(config=config)
    train_epoch = 1920
    assert (train_epoch in config.max_epochs)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
    except FileNotFoundError as e:
        assert FileNotFoundError(
            "Error {}; no valid trained model found; train before initiating experiment\n".format(e))
    # cleanup_experiment()

    if config.has_cuda:
        # Sampling is sequential, so only single-machine, single-GPU/CPU
        device = 0
    else:
        device = torch.device("cpu")
    data_shape = (config.dataSize, config.ts_length, 1)
    fBm = torch.from_numpy(
        np.load("/Users/marcos/GitHubRepos/FractionalBrownianMotion/data/fBn_samples_H07_T256.npy").cumsum(axis=1)[
        :config.dataSize, :])
    fBm = fBm.unsqueeze(-1).to(torch.float32).to(device)
    assert (fBm.shape == data_shape)
    scoreModel.eval()
    scoreModel.to(device)
    features = []
    with torch.no_grad():
        samples = torch.zeros(size=(data_shape[0], 1, data_shape[-1])).to(device)
        for t in range(config.ts_length):
            print("Sampling at real time {}\n".format(t + 1))
            if t == 0:
                output, (h, c) = scoreModel.rnn(samples, None)
            else:
                output, (h, c) = scoreModel.rnn(fBm[:, [t - 1], :], (h, c))
            features.append(output.permute(1, 0, 2))
    feature_df = torch.concat(features, dim=0).cpu()
    assert (feature_df.shape == (config.ts_length, config.dataSize, 40))
    feature_df = pd.concat({i: pd.DataFrame(feature_df[i, :, :]) for i in tqdm(range(config.ts_length))})
    feature_data_path = config.experiment_path.replace("results/", "results/feature_data/") + "_NEp{}_Exact".format(
        train_epoch).replace(".", "") + ".parquet.gzip"
    feature_df.info()
    print(feature_df)
    feature_df.to_parquet(feature_data_path, compression="gzip")


if __name__ == "__main__":
    exact_fBm_features()
