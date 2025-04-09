import os

import numpy as np
import torch

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fBiPot.recursive_LSTM_PostMeanScore_fBiPot_T256_H05_tl_110data import \
    get_config as get_config
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.drift_evaluation_functions import LSTM_1D_drifts

if __name__ == "__main__":
    config = get_config()
    for Nepoch in config.max_epochs:
        try:
            print(f"Starting Epoch {Nepoch}\n")
            # Fix the number of training epochs and training loss objective loss
            PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters)
            PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
            final_vec_mu_hats = LSTM_1D_drifts(PM=PM, config=config)
            type = "PM"
            assert (type in config.scoreNet_trained_path)
            if "_ST_" in config.scoreNet_trained_path:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_ST_fBiPot_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                    ".", "")
            else:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fBiPot_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.quartic_coeff}a_{config.quad_coeff}b_{config.const}c_{config.residual_layers}ResLay_{config.loss_factor}LFac_BetaMax{config.beta_max:.1e}").replace(
                    ".", "")
            print(f"Save path:{save_path}\n")
            assert config.ts_dims == 1
            np.save(save_path + "_muhats.npy", final_vec_mu_hats)
        except FileNotFoundError as e:
            print(e)
            continue
