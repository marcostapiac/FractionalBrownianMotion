import ml_collections
import torch

from configs import project_config


def get_config():
    """ Training hyperparameters for VP SDE model on circle samples with 0.03 noise factor """

    config = ml_collections.ConfigDict()

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Data set parameters
    config.ts_length = 2
    config.cnoise = 0.03
    config.data_path = project_config.ROOT_DIR + "data/noisy_circle_samples.npy"

    # Training hyperparameters
    config.train_eps = 1./config.max_diff_steps
    config.max_diff_steps = 1000
    config.end_diff_time = 1.
    config.save_freq = 50
    config.lr = 1e-3
    config.max_epochs = [20000]config.max_epochs = 200
    config.batch_size = 256
    config.hybrid = True
    # Diffusion hyperparameters
    config.beta_max = 20.
    config.beta_min = 0.1

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
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_noisy_circle_VPSDE_model_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_BetaMax{:.1e}_BetaMin{:.1e}_TembDim{}_EncShapes{}".format(
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.beta_max, config.beta_min,
config.temb_dim,
config.residual_layers, config.residual_channels, config.diff_hidden_size, config.hybrid, config.weightings, config.t0, config.deltaT,
config.quad_coeff, config.sin_coeff, config.sin_space_scale, config.mlp_hidden_dims, config.condupsampler_length, config.tdata_mult).replace(".", "")

    tsmFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_noisy_circle_VPSDE_model_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_BetaMax{:.1e}_BetaMin{:.1e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts".format(
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.beta_max, config.beta_min,
        config.temb_dim,
        config.residual_layers, config.residual_channels, config.diff_hidden_size, config.hybrid,
        config.weightings).replace(".", "")

    config.model_choice = "TSM"
    config.scoreNet_trained_path = tsmFileName if config.model_choice == "TSM" else mlpFileName
    config.model_parameters = [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.residual_layers,
                               config.residual_channels, config.dialation_length] \
        if config.model_choice == "TSM" else [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.ts_dims, config.mlp_hidden_dims,
                                      config.condupsampler_length, config.residual_layers,
                       config.residual_channels, config.dialation_length]

    # Snapshot filepath
    config.scoreNet_snapshot_path = config.scoreNet_trained_path.replace("trained_models/", "snapshots/")

    # Sampling hyperparameters
    config.early_stop_idx = 0
    config.sample_eps = 1e-3
    config.max_lang_steps = 0
    config.snr = 0.
    config.predictor_model = "Ancestral"  # vs "euler-maryuama"
    config.corrector_model = "VP"  # vs "VE"

    # Experiment evaluation parameters
    config.dataSize = 100000
    config.num_runs = 10
    config.plot = False
    config.image_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                             "pngs/")
    config.exp_keys = ["Mean Abs Percent Diff", "Cov Abs Percent Diff", "Marginal p-vals", "True Inner/Outer",
                       "Gen Inner/Outer"]
    config.experiment_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                                  "experiments/results/")

    # LSTM parameters
    config.test_lstm = False

    return config
