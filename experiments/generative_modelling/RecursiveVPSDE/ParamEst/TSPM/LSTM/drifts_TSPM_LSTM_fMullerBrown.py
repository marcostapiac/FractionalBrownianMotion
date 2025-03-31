import os

import numpy as np
import torch

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fMullerBrown.recursive_LSTM_PostMeanScore_MullerBrown_T256_H05_tl_110data import \
    get_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from utils.drift_evaluation_functions import find_LSTM_feature_vectors_multiDTS

config = get_config()


def LSTM_2D_drifts(PM, config):
    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")
    PM = PM.to(device)

    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)
    ts_step = config.deltaT

    num_taus = 100

    num_diff_times = config.max_diff_steps
    Ndiff_discretisation = config.max_diff_steps
    diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                     steps=Ndiff_discretisation).to(device)

    numXs = 25
    minx = -1.
    maxx = -0.9
    Xs = np.linspace(minx, maxx, numXs)
    miny = 1.
    maxy = 1.1
    Ys = np.linspace(miny, maxy, numXs)
    Xs, Ys = np.meshgrid(Xs, Ys)
    Xs = np.column_stack([Xs.ravel(), Ys.ravel()])
    Xshape = Xs.shape[0]
    features = find_LSTM_feature_vectors_multiDTS(Xs=Xs, score_model=PM, device=device, config=config)
    num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in Xs}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    final_vec_mu_hats = np.zeros(
        (Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims

    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)
    # ts = []
    es = 1
    # mu_hats_mean = np.zeros((tot_num_feats, num_taus))
    # mu_hats_std = np.zeros((tot_num_feats, num_taus))
    difftime_idx = num_diff_times - 1
    PM.eval()
    while difftime_idx >= num_diff_times - es:
        d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)

        with torch.no_grad():
            vec_predicted_score = PM.forward(inputs=vec_Z_taus, times=vec_diff_times, conditioner=vec_conditioner,
                                             eff_times=vec_eff_times)
        vec_scores, vec_drift, vec_diffParam = diffusion.get_conditional_reverse_diffusion(x=vec_Z_taus,
                                                                                           predicted_score=vec_predicted_score,
                                                                                           diff_index=torch.Tensor(
                                                                                               [int((
                                                                                                       num_diff_times - 1 - difftime_idx))]).to(
                                                                                               device),
                                                                                           max_diff_steps=Ndiff_discretisation)
        # assert np.allclose((scores- predicted_score).detach(), 0)
        beta_taus = torch.exp(-0.5 * eff_times[0, 0, 0]).to(device)
        sigma_taus = torch.pow(1. - torch.pow(beta_taus, 2), 0.5).to(device)
        final_mu_hats = (vec_Z_taus / (ts_step * beta_taus)) + ((
                                                                        (torch.pow(sigma_taus, 2) + (
                                                                                torch.pow(beta_taus, 2) * ts_step)) / (
                                                                                ts_step * beta_taus)) * vec_scores)

        assert (final_mu_hats.shape == (num_taus * tot_num_feats, 1, config.ts_dims))
        A = final_mu_hats.reshape((num_taus, tot_num_feats, config.ts_dims))
        split_tensors = torch.split(A, list(num_feats_per_x.values()), dim=1)
        # Compute means along the column dimension
        means = torch.stack([t.mean(dim=1) for t in split_tensors], dim=1)
        assert (means.shape == (num_taus, Xshape, config.ts_dims))
        # print(vec_Z_taus.shape, vec_scores.shape)
        final_vec_mu_hats[:, difftime_idx, :] = means.permute((1, 0, 2)).cpu().numpy()
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
    return final_vec_mu_hats[:, -es:, :, :]


if __name__ == "__main__":
    config = get_config()
    for Nepoch in config.max_epochs:
        try:
            print(f"Starting Epoch {Nepoch}\n")
            # Fix the number of training epochs and training loss objective loss
            PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters)
            PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
            final_vec_mu_hats = LSTM_2D_drifts(PM=PM, config=config)
            type = "PM"
            assert (type in config.scoreNet_trained_path)
            if "_ST_" in config.scoreNet_trained_path:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_ST_fMullerBrown_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                    ".", "")
            else:
                save_path = (
                        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fMullerBrown_DriftEvalExp_{Nepoch}Nep_{config.t0}t0_{config.deltaT:.3e}dT_{config.residual_layers}ResLay_{config.loss_factor}LFac").replace(
                    ".", "")
            print(f"Save path:{save_path}\n")
            assert config.ts_dims == 2
            np.save(save_path + "_muhats.npy", final_vec_mu_hats)
        except FileNotFoundError as e:
            print(e)
            continue
