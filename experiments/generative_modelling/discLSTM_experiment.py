import numpy as np
import torch

from src.evaluation_pipeline.classes.DiscriminativeLSTM.ClassDiscriminativeLSTM import DiscriminativeLSTM
from src.evaluation_pipeline.data_processing import train_and_save_discLSTM
from src.generative_modelling.data_processing import reverse_sampling
from src.generative_modelling.models.ClassVESDEDiffusion import VESDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassNaiveMLP import NaiveMLP
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassTimeSeriesScoreMatching import \
    TimeSeriesScoreMatching
from utils.data_processing import test_discLSTM
from utils.math_functions import generate_fBm

if __name__ == "__main__":
    from configs.VESDE.fBm_T32_H07 import get_config

    config = get_config()

    scoreModel = TimeSeriesScoreMatching(*config.model_parameters) if config.model_choice == "TSM" else NaiveMLP(
        *config.model_parameters)
    diffusion = VESDEDiffusion(stdMax=config.std_max, stdMin=config.std_min)

    try:
        scoreModel.load_state_dict(torch.load(config.scoreNet_trained_path))
    except (FileNotFoundError) as e:
        print("Please train a valid diffusion model before beginning train-synthetic-test-real evaluation\n")

    # Now generate synthetic samples
    model = DiscriminativeLSTM(ts_dim=1)
    dataSize = min(10 * sum(p.numel() for p in model.parameters() if p.requires_grad), 2000000) // 100
    try:
        model.load_state_dict(torch.load(config.disc_lstm_trained_path))
    except FileNotFoundError as e:
        # Train on synthetic
        rng = np.random.default_rng()
        synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel, data_shape=(dataSize, config.timeDim),
                                     config=config)
        original = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, rng=rng)
        train_and_save_discLSTM(org_data=original, synth_data=synthetic.cpu().numpy(), config=config, model=model)

    # Now test on real (but also synthetic to compare MAEs)
    rng = np.random.default_rng()
    synthetic = reverse_sampling(diffusion=diffusion, scoreModel=scoreModel, data_shape=(dataSize, config.timeDim),
                                 config=config)
    original = generate_fBm(H=config.hurst, T=config.timeDim, S=dataSize, rng=rng)
    test_discLSTM(original_data=original, synthetic_data=synthetic.cpu().numpy(), config=config, model=model)
