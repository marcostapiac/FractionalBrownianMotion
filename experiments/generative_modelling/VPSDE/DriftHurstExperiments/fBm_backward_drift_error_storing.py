import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import run_fBm_backward_drift_experiment


def run(config: ConfigDict):
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_Nepochs" + str(
                                                                                                            config.max_epochs)))

    drift_errors, score_only_drift_errors = run_fBm_backward_drift_experiment(dataSize=10000, diffusion=diffusion,
                                                                              scoreModel=scoreModel,
                                                                              rng=rng,
                                                                              config=config)

    drift_data_path = config.experiment_path.replace("results/", "results/drift_data/") + "_DriftErrorsTS_Nepochs{}".format(
        config.max_epochs).replace(
        ".", "")

    score_only_drift_data_path = drift_data_path.replace("DriftErrorsTS", "ScoreOnlyDriftErrorsTS")

    pd.DataFrame(data=drift_errors.numpy()).to_csv(drift_data_path + ".csv.gzip", compression="gzip", index=True)
    pd.DataFrame(data=score_only_drift_errors.numpy()).to_csv(score_only_drift_data_path + ".csv.gzip",
                                                              compression="gzip", index=True)


if __name__ == "__main__":
    # Data parameters
    from configs.VPSDE.fBm_T256_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)
    assert (config.early_stop_idx == 0)
    run(config)
