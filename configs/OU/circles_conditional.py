import ml_collections
import torch

from configs import project_config


def get_config():
    """ Training hyperparameters for conditional OUSDE SDE model on circle samples with 0.03 noise factor """

    config = ml_collections.ConfigDict()

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Data set parameters
    config.timeDim = 2
    config.cnoise = 0.03
    config.data_path = project_config.ROOT_DIR + "data/noisy_circle_samples.npy"

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Training hyperparameters
    config.train_eps = 1e-3
    config.max_diff_steps = 1000
    config.end_diff_time = 1.
    config.save_freq = 50
    config.lr = 1e-3
    config.max_epochs = 400
    config.batch_size = 256

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
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_noisy_circle_OUSDE_conditional_model_Ndiff{}_Tdiff{}_trainEps{:.0e}_TembDim{}_EncShapes{}".format(
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.temb_dim,
        config.enc_shapes)

    tsmFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_noisy_circle_OUSDE_conditional_model_Ndiff{}_Tdiff{}_trainEps{:.0e}_DiffEmbSize{}_ResidualLayers{}_ResChan{}_DiffHiddenSize{}".format(
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.temb_dim,
        config.residual_layers, config.residual_channels, config.diff_hidden_size)

    config.model_choice = "TSM"
    config.filename = tsmFileName if config.model_choice == "TSM" else mlpFileName
    config.model_parameters = [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.residual_layers,
                               config.residual_channels, config.dialation_length] \
        if config.model_choice == "TSM" else [config.temb_dim, config.max_diff_steps, config.timeDim, config.enc_shapes,
                                              config.dec_shapes]

    # Snapshot filepath
    config.snapshot_path = config.filename.replace("trained_models/", "snapshots/")

    # Sampling hyperparameters
    config.sample_eps = 1e-3
    config.max_lang_steps = 0
    config.snr = 0.
    config.predictor_model = "euler-maruyama"  # vs "ancestral"
    config.corrector_model = "OU"

    return config