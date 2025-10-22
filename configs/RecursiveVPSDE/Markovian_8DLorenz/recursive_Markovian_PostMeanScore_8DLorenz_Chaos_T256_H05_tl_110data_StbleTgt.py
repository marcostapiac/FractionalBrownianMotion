import ml_collections
import torch

from configs import project_config


def get_config():
    config = ml_collections.ConfigDict()

    # Experiment environment parameters
    config.has_cuda = torch.cuda.is_available()

    # Data set parameters
    config.hurst = 0.5
    # Classic stable behaviour (forcing_const < 1)
    config.ndims = 8
    config.forcing_const = 1.25
    config.diffusion = 1.
    config.initState = [0.0, 0.214, 4.29, 0.643, 0.857, 1.071, 12.86, 1.5]#([0., .5, 1., 1.5, 0., .5, 1., 1.5,])
    config.ts_length = 256
    config.t0 = 0.
    config.deltaT = 1. / (256)
    config.t1 = config.deltaT*config.ts_length
    config.data_path = project_config.ROOT_DIR + "data/{}DLnz_samples_t0{:g}_dT{:.3e}_T{}_{}FConst_{}Diff".format(config.ndims,
        config.t0, config.deltaT, config.ts_length, config.forcing_const, config.diffusion).replace(
        ".", "") + ".npy"


    # Training hyperparameters
    config.max_diff_steps = 10000
    config.train_eps = 1./config.max_diff_steps  # 1000 * max(int(np.log2(config.ts_length) - 1), 1)
    config.end_diff_time = 1.
    config.save_freq = 2
    config.lr = 7.5e-5
    config.max_epochs = [3000]
    config.ref_batch_size = 2048
    config.batch_size = 256
    config.chunk_size = 512
    config.feat_thresh = 1./100.
    config.isfBm = True
    config.isUnitInterval = True
    config.hybrid = True
    config.weightings = True
    config.tdata_mult = 110
    config.ts_dims = config.ndims
    config.loss_factor = 2

    # Diffusion hyperparameters
    config.beta_max = 20.
    config.beta_min = 0.  # 0.0001

    # Universal Architecture Parameters
    config.temb_dim = 64
    config.residual_layers = 10
    config.residual_channels = 8
    config.diff_hidden_size = 64
    config.dialation_length = 10
    config.enforce_fourier_mean_reg = True


    # MLP Architecture parameters 
    config.mlp_hidden_dims = 16
    config.condupsampler_length = 20

    # TSM Architecture parameters
    config.lstm_hiddendim = 20
    config.lstm_numlay = 1
    config.lstm_inputdim = config.ts_dims
    config.lstm_dropout = 0.
    assert ((config.lstm_dropout == 0. and config.lstm_numlay == 1) or (
            config.lstm_dropout > 0 and config.lstm_numlay > 1))

    # Model filepath
    mlpFileName = project_config.ROOT_DIR + "src/generative_modelling/trained_models/trained_rec_PM_ST_{:.3f}FTh_MLP_{}LFac_{}DLnz_{:.2e}FConst_VPSDE_T{}_Ndiff{}_Tdiff{:.3e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts_t0{:g}_dT{:.3e}_MLP_H{}_CUp{}_tl{}".format(
        config.feat_thresh, config.loss_factor, config.ndims, config.forcing_const,
        config.ts_length,
        config.max_diff_steps, config.end_diff_time, 
        config.temb_dim,
        config.residual_layers, config.residual_channels, config.diff_hidden_size, config.hybrid, config.weightings,
        config.t0, config.deltaT,
        config.mlp_hidden_dims, config.condupsampler_length, config.tdata_mult, config.tdata_mult).replace(".", "")

    config.model_choice = "MLP"
    config.scoreNet_trained_path =  mlpFileName
    config.model_parameters = [config.max_diff_steps, config.temb_dim, config.diff_hidden_size, config.ts_dims, config.mlp_hidden_dims,
                                      config.condupsampler_length, config.residual_layers,
                       config.residual_channels, config.dialation_length]

    # Snapshot filepath
    config.scoreNet_snapshot_path = config.scoreNet_trained_path.replace("trained_models/", "snapshots/")
    config.resource_logging_path = config.scoreNet_trained_path.replace("trained_models/", "resource_logging/") + ".json"
    config.nadaraya_resource_logging_path = config.scoreNet_trained_path.replace("trained_models/", "resource_logging/") + ".json".replace("trained_rec_PM_ST_{:.3f}FTh_MLP_{}LFac_", "").replace("_Ndiff{}_Tdiff{:.3e}_DiffEmbSz{}_ResLay{}_ResChan{}_DiffHdnSz{}_{}Hybd_{}Wghts_", "_").replace("_MLP_H{}_CUp{}", "") + ".json"

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
