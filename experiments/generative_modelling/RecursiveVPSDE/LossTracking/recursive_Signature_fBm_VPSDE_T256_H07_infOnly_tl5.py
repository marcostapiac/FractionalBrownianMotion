import numpy as np
import pandas as pd
import torch

from src.generative_modelling.data_processing import recursive_signature_reverse_sampling
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalSignatureTSScoreMatching import \
    ConditionalSignatureTSScoreMatching
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP

if __name__ == "__main__":
    # Data parameters
    from configs.RecursiveVPSDE.recursive_Signature_fBm_T256_H07_tl_5data import get_config

    config = get_config()
    assert (0 < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    assert (config.tdata_mult == 5)
    print(config.scoreNet_trained_path, config.dataSize)
    rng = np.random.default_rng()
    scoreModel = ConditionalSignatureTSScoreMatching(
        *config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    for train_epoch in [960]:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(train_epoch)))
        final_paths = recursive_signature_reverse_sampling(diffusion=diffusion, scoreModel=scoreModel,
                                                           data_shape=(config.dataSize, config.ts_length, 1),
                                                           config=config)
        df = pd.DataFrame(final_paths)
        df.index = pd.MultiIndex.from_product(
            [["Final Time Samples"], [i for i in range(config.dataSize)]])
        df.to_csv(config.experiment_path + "_NEp{}.csv.gzip".format(train_epoch), compression="gzip")
