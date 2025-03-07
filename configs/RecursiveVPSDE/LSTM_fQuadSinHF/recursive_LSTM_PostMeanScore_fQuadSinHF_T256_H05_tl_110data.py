import ml_collections
import torch

from configs import project_config


def get_config():
    config = ml_collections.ConfigDict()

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()
    # Data set parameters
    config.hurst = 0.5
    config.quad_coeff = .5
    assert(config.quad_coeff > 0.)
    config.sin_coeff = 1./25.
    config.sin_space_scale = 25.
    config.diffusion = 1.
    config.initState = 0.
    config.ts_length = 256
    config.t0 = 0.
    config.deltaT = 1./(64)
    config.t1 = config.deltaT*config.ts_length

    config.data_path = project_config.ROOT_DIR + "data/fQuadSinHF_samples_t0{:g}_dT{:.3e}_T{}_{}a_{}b_{}c_{}Diff_{}Init".format(
        config.t0, config.deltaT, config.ts_length, config.quad_coeff, config.sin_coeff, config.sin_space_scale,
        config.diffusion, config.initState).replace(
        ".", "") + ".npy"

    # Training hyperparameters
    config.max_diff_steps = 10000
    config.train_eps = 1./config.max_diff_steps  # 1000 * max(int(np.log2(config.ts_length) - 1), 1)
    config.end_diff_time = 1.
    config.save_freq = 50
    config.lr = 1e-3
    config.max_epochs = [60, 100, 150, 300, 960, 1440, 1920, 2920, 6920, 7190, 8190, 9700, 12920, 14920, 16920, 18920, 20920, 22920]
    config.batch_size = 256
    config.isfBm = True
    config.isUnitInterval = True
    config.hybrid = True
    config.weightings = True
    config.tdata_mult = 110
    config.ts_dims = 1
    config.loss_factor = 0

    # Diffusion hyperparameters
    config.beta_max = 20.
    config.beta_min = 0.  # 0.0001

    # MLP Architecture parameters
    config.temb_dim = 64
    config.enc_shapes = [8, 16, 32]
    config.dec_shapes = config.enc_shapes[::-1]

    # TSM Architecture parameters
    config.residual_layers = 10
    config.residual_channels = 8
    config.diff_hidden_size = 64
    config.dialation_length = 10
    config.lstm_hiddendim = 20
    config.lstm_numlay = 1
    config.lstm_inputdim = config.ts_dims
    config.lstm_dropout = 0.
    assert ((config.lstm_dropout == 0. and config.lstm_numlay == 1) or (
            config.lstm_dropout > 0 and config.lstm_numlay > 1))

    # Model filepath
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_rec_PM_MLP_{}_incs_fQuadSinHF_VPSDE_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_BetaMax{:.1e}_BetaMin{:.1e}_TembDim{}_EncShapes{}_tl110".format(
        not config.isfBm, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.beta_max, config.beta_min,
        config.temb_dim,
        config.enc_shapes).replace(".", "")

    tsmFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_rec_PM_TSM_{}_incs_fQuadSinHF_VPSDE_H{:.1e}_T{}_Ndiff{}_Tdiff{:.3e}_trainEps{:.0e}_BetaMax{:.1e}_BetaMin{:.1e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts_t0{:g}_dT{:.3e}_{}a_{}b_{}c_LSTM_H{}_Nly{}_tl{}".format(
        not config.isfBm, config.hurst,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, config.train_eps, config.beta_max, config.beta_min,
        config.temb_dim,
        config.residual_layers, config.residual_channels, config.diff_hidden_size, config.hybrid, config.weightings, config.t0, config.deltaT,
        config.quad_coeff, config.sin_coeff, config.sin_space_scale, config.lstm_hiddendim, config.lstm_numlay, config.tdata_mult).replace(".", "")

    config.model_choice = "TSM"
    config.scoreNet_trained_path = tsmFileName if config.model_choice == "TSM" else mlpFileName
    config.model_parameters = [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.lstm_hiddendim,
                               config.lstm_numlay, config.lstm_inputdim, config.lstm_dropout, config.residual_layers,
                               config.residual_channels, config.dialation_length] \
        if config.model_choice == "TSM" else [config.temb_dim, config.max_diff_steps, config.ts_length,
                                              config.enc_shapes,
                                              config.dec_shapes]

    # Snapshot filepath
    config.scoreNet_snapshot_path = config.scoreNet_trained_path.replace("trained_models/", "snapshots/")

    # Sampling hyperparameters
    config.early_stop_idx = 0
    config.sample_eps = config.train_eps
    if config.hybrid: assert (config.sample_eps == config.train_eps)
    config.max_lang_steps = 0
    config.snr = 0.
    config.predictor_model = "CondAncestral"
    config.corrector_model = "VP"  # vs "VE" vs "OUSDE"
    config.param_time = 900

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
        "src/generative_modelling/trained_models/trained_",
        "src/evaluation_pipeline/trained_models/rec_trained_discLSTM_")
    config.disc_lstm_snapshot_path = config.disc_lstm_trained_path.replace("trained_models/", "snapshots/")
    config.pred_lstm_trained_path = config.disc_lstm_trained_path.replace("disc", "pred")
    config.pred_lstm_snapshot_path = config.disc_lstm_snapshot_path.replace("disc", "pred")

    return config
