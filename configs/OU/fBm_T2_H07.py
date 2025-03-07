import ml_collections
import numpy as np
import torch

from configs import project_config


def get_config():
    """ Training hyperparameters for OUSDE SDE model on 2-dimensional Fractional Brownian Motion with Hurst parameter 0.7"""

    config = ml_collections.ConfigDict()

    # Environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Data set parameters
    config.hurst = 0.7
    config.ts_length = 2
    config.data_path = project_config.ROOT_DIR + "data/fBn_samples_H{}_T{}.npy".format(
        str(config.hurst).replace(".", ""), config.ts_length)

    # Training hyperparameters
    config.train_eps = 1./config.max_diff_steps
    config.max_diff_steps = 1000 * max(int(np.log2(config.ts_length) - 1), 1)
    config.end_diff_time = 1.
    config.save_freq = 50
    config.lr = 1e-3
    config.max_epochs = 1000
    config.batch_size = 256
    config.isfBm = True
    config.isUnitInterval = True
    config.hybrid = True
    config.weightings = True

    # MLP Architecture parameters
    config.temb_dim = 32
    config.enc_shapes = [8, 16, 32]
    config.dec_shapes = config.enc_shapes[::-1]

    # TSM Architecture parameters
    config.residual_layers = 10
    config.residual_channels = 8
    config.diff_hidden_size = 32
    config.dialation_length = 10

    # Model filepath
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_{}LFac_fBm_OUSDE_model_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_TembDim{}_EncShapes{}".format(
        config.loss_factor, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.temb_dim,
        config.enc_shapes).replace(".", "")

    tsmFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_{}LFac_fBm_OUSDE_model_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts".format(
        config.loss_factor, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.temb_dim,
        config.residual_layers, config.residual_channels, config.diff_hidden_size, config.hybrid,
        config.weightings).replace(".", "")

    config.model_choice = "TSM"
    config.scoreNet_trained_path = tsmFileName if config.model_choice == "TSM" else mlpFileName
    config.model_parameters = [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.residual_layers,
                               config.residual_channels, config.dialation_length] \
        if config.model_choice == "TSM" else [config.temb_dim, config.max_diff_steps, config.ts_length,
                                              config.enc_shapes,
                                              config.dec_shapes]

    # Snapshot filepath
    config.scoreNet_snapshot_path = config.scoreNet_trained_path.replace("trained_models/", "snapshots/")

    # Sampling hyperparameters
    config.early_stop_idx = 0
    config.sample_eps = 1e-3
    config.max_lang_steps = 0
    config.snr = 0.
    config.predictor_model = "euler-maruyama"  # vs "euler-maryuama"
    config.corrector_model = "OU"

    # Experiment evaluation parameters
    config.dataSize = 100000
    config.num_runs = 10
    config.unitInterval = True
    config.annot_heatmap = True
    config.plot = True
    config.permute_test = False
    config.image_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                             "pngs/")
    config.exp_keys = ["Mean Abs Percent Diff", "Cov Abs Percent Diff", "Chi2 Lower", "Chi2 Upper", "Chi2 True Stat",
                       "Chi2 Synthetic Stat", "Marginal p-vals", "Original MAE", "Synthetic MAE", "Original Disc Score",
                       "Synthetic Disc Score"]
    config.experiment_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                                  "experiments/results/")

    return config
