import os

import numpy as np
import torch

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fMullerBrown.recursive_LSTM_PostMeanScore_MullerBrown_T256_H05_tl_110data import \
    get_config
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.drift_evaluation_functions import LSTM_2D_drifts

config = get_config()


if __name__ == "__main__":
    config = get_config()
    for Nepoch in config.max_epochs:
        try:
            print(f"Starting Epoch {Nepoch}\n")
            # Fix the number of training epochs and training loss objective loss
            PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters)
            PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
            final_vec_mu_hats = LSTM_2D_drifts(PM=PM, config=config)
            type = "PM"
            assert (type in config.scoreNet_trained_path)
            if "_ST_" in config.scoreNet_trained_path:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_ST_fMullerBrown_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                    ".", "")
            else:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fMullerBrown_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                    ".", "")
            print(f"Save path:{save_path}\n")
            assert config.ts_dims == 2
            np.save(save_path + "_muhats.npy", final_vec_mu_hats)
        except FileNotFoundError as e:
            print(e)
            continue
