import ml_collections
import torch

from configs import project_config


def get_config():
    config = ml_collections.ConfigDict()

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Data set parameters
    config.hurst = 0.7
    config.ts_length = 2 ** 10
    config.data_path = project_config.ROOT_DIR + "data/fBn_samples_H{}_T{}.npy".format(
        str(config.hurst).replace(".", ""), config.ts_length)

    # Training hyperparameters
    config.train_eps = 1./config.max_diff_steps
    config.max_diff_steps = 20000  # 1000*max(int(np.log(config.ts_length-1, 1)))
    config.end_diff_time = 1.
    config.save_freq = 50
    config.lr = 1e-3
    config.max_epochs = [20000]config.max_epochs = 7060
    config.batch_size = 256
    config.isfBm = True
    config.isUnitInterval = True
    config.hybrid = True
    config.weightings = False

    config.std_max = 90.
    config.std_min = 0.01

    # MLP Architecture parameters
    config.temb_dim = 64
    config.enc_shapes = [8, 16, 32]
    config.dec_shapes = config.enc_shapes[::-1]

    # TSM Architecture parameters
    config.residual_layers = 10
    config.residual_channels = 8
    config.diff_hidden_size = 256
    config.dialation_length = 10

    # Model filepath
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_MLP_{}LFac_fBm_VESDE_model_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_StdMax{:.4e}_StdMin{:.4e}_TembDim{}_EncShapes{}".format(
        config.loss_factor, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.std_max, config.std_min, config.temb_dim,
        config.enc_shapes).replace(".", "")

    tsmFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_TSM_{}LFac_fBm_VESDE_model_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_StdMax{:.3e}_StdMin{:.4e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts".format(
        config.loss_factor, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.std_max, config.std_min, config.temb_dim,
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
    config.sample_eps = 1e-5
    if config.hybrid: assert (config.sample_eps == config.train_eps)
    config.max_lang_steps = 0
    config.snr = 0.01
    config.predictor_model = "Ancestral"  # vs "euler-maryuama"
    config.corrector_model = "VE"  # vs "VE"

    # Experiment evaluation parameters
    config.dataSize = 40000
    config.num_runs = 20
    config.unitInterval = True
    config.plot = False
    config.annot_heatmap = False
    config.permute_test = False
    config.image_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                             "pngs/")
    config.exp_keys = ["Mean Abs Percent Diff", "Cov Abs Percent Diff", "Chi2 Lower", "Chi2 Upper", "Chi2 True Stat",
                       "Chi2 Synthetic Stat", "Marginal p-vals", "Original Pred Score", "Synthetic Pred Score",
                       "Original Disc Score", "Synthetic Disc Score", "True Hurst Estimates",
                       "Synthetic Hurst Estimates"]

    config.experiment_path = config.scoreNet_trained_path.replace("src/generative_modelling/trained_models/trained_",
                                                                  "experiments/results/")

    # LSTM parameters
    config.test_pred_lstm = False
    config.test_disc_lstm = False
    config.lookback = 10
    config.pred_lstm_max_epochs = 700
    config.disc_lstm_max_epochs = 5000
    config.pd_lstm_batch_size = 128
    config.disc_lstm_trained_path = config.scoreNet_trained_path.replace(
        "src/generative_modelling/trained_models/trained_", "src/evaluation_pipeline/trained_models/trained_discLSTM_")
    config.disc_lstm_snapshot_path = config.disc_lstm_trained_path.replace("trained_models/", "snapshots/")
    config.pred_lstm_trained_path = config.disc_lstm_trained_path.replace("disc", "pred")
    config.pred_lstm_snapshot_path = config.disc_lstm_snapshot_path.replace("disc", "pred")

    return config
