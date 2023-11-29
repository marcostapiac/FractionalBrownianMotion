import numpy as np
import pandas as pd
import torch
from ml_collections import ConfigDict

from configs import project_config
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.experiment_evaluations import run_fBm_backward_drift_experiment


def run(config: ConfigDict):
    rng = np.random.default_rng()
    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)
    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path + "_Nepochs" + str(config.max_epochs)))
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Error {}; no valid trained model found; train model {} before running experiment\n".format(e,
                                                                                                        config.scoreNet_trained_path + "_Nepochs" + str(
                                                                                                            config.max_epochs)))

    drift_errors, score_only_drift_errors = run_fBm_backward_drift_experiment(dataSize=5000, diffusion=diffusion,
                                                                              scoreModel=scoreModel,
                                                                              rng=rng,
                                                                              config=config)

    drift_pic_path = project_config.ROOT_DIR + "experiments/results/drift_data_and_plots/DriftErrorsTS_fBm_H{:.3e}_T{}_Ndiff{}_Tdiff{:.3e}_StdMax{:.4e}_StdMin{:.4e}_Nepochs{}".format(
        config.hurst, config.timeDim, config.max_diff_steps, config.end_diff_time, config.std_max,
        config.std_min, config.max_epochs).replace(
        ".", "")

    score_only_drift_pic_path = drift_pic_path.replace("DriftErrorsTS", "ScoreOnlyDriftErrorsTS")

    pd.DataFrame(data=drift_errors.numpy()).to_csv(drift_pic_path + ".csv.gzip", compression="gzip", index=True)
    pd.DataFrame(data=score_only_drift_errors.numpy()).to_csv(score_only_drift_pic_path + ".csv.gzip",
                                                              compression="gzip", index=True)


if __name__ == "__main__":
    # Data parameters
    from configs.VESDE.fBm_T32_H07 import get_config

    config = get_config()
    assert (0. < config.hurst < 1.)

    run(config)
