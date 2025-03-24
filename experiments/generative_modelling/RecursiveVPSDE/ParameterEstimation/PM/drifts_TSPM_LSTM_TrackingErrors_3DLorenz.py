import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from configs.RecursiveVPSDE.LSTM_3DLorenz.recursive_LSTM_PostMeanScore_3DLorenz_T256_H05_tl_110data import \
    get_config
from configs import project_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from tqdm import tqdm


def find_LSTM_feature_vectors(Xs, PM, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float).to(device)

    def process_single_threshold(x, dX_global):
        diff = sim_data_tensor - x.reshape(1, -1)  # shape: (M, N, D)
        # tensor_norm = sim_data_tensor / sim_data_tensor.norm(dim=-1, keepdim=True)
        # candidate_norm = x / x.norm(dim=0, keepdim=True)  # (D, 1)
        # diff = (tensor_norm @ candidate_norm).squeeze(-1) # Result: (M, N)
        # mask = diff >= dX_global
        diff = diff.norm(dim=-1)  # Result: (M, N)
        mask = diff <= np.arccos(dX_global)
        thresh = dX_global
        while torch.sum(mask) == 0:
            thresh = np.cos(np.arccos(thresh)*2)
            mask = diff <= np.arccos(thresh)
        assert (torch.sum(mask) > 0)

        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)
        sequences = []
        js = []
        for idx in indices:
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1, :]
            sequences.append(seq)
            js.append(len(seq))
        outputs = []
        PM.eval()
        sequences = sequences[:100]
        js = js[:100]
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan).to(device)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            # padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = PM.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    features_Xs = {}
    dX_global = np.cos(1. / 5000)
    for i in range(Xs.shape[0]):
        x = Xs[i, :].reshape(-1, 1)
        x_val, out = process_single_threshold(torch.tensor(x).to(device, dtype=torch.float), dX_global)
        assert (len(out) > 0)
        print(out.shape)
        features_Xs[tuple(x.squeeze().tolist())] = out
    return features_Xs


def true_drift(prev, num_paths, config):
    assert (prev.shape == (num_paths, config.ndims))
    drift_X = np.zeros((num_paths, config.ndims))
    drift_X[0] = config.ts_sigma * (prev[1] - prev[0])
    drift_X[1] = (prev[0] * (config.ts_rho - prev[2]) - prev[1])
    drift_X[2] = (prev[0] * prev[1] - config.ts_beta * prev[2])
    return drift_X[:, np.newaxis, :]


def multivar_score_based_LSTM_drift(score_model, num_diff_times, diffusion, num_paths, prev, ts_step, config,
                                    device):
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    features = find_LSTM_feature_vectors(Xs=prev, PM=PM, device=device, config=config)
    num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in prev}
    # list_num_feats_per_x = list(num_feats_per_x.values())
    tot_num_feats = np.sum(list(num_feats_per_x.values()))
    features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
    assert (features_tensor.shape[0] == tot_num_feats)
    vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)

    diffusion_times = torch.linspace(config.sample_eps, 1., config.max_diff_steps)
    difftime_idx = Ndiff_discretisation - 1
    while difftime_idx >= Ndiff_discretisation - num_diff_times:
        d = diffusion_times[difftime_idx].to(device)
        diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
        eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
        vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
        vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
        vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(
            num_taus * tot_num_feats,
            1, -1)
        score_model.eval()
        with torch.no_grad():
            vec_predicted_score = score_model.forward(times=vec_diff_times, eff_times=vec_eff_times,
                                                      conditioner=vec_conditioner, inputs=vec_Z_taus)
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
        assert (means.shape == (num_taus, num_paths, config.ts_dims))

        # print(vec_Z_taus.shape, vec_scores.shape)
        vec_z = torch.randn_like(vec_drift).to(device)
        vec_Z_taus = vec_drift + vec_diffParam * vec_z
        difftime_idx -= 1
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy()


if __name__ == "__main__":
    config = get_config()
    assert ("3DLnz" in config.data_path)

    print("Beta Min : ", config.beta_min)
    if config.has_cuda:
        device = int(os.environ["LOCAL_RANK"])
    else:
        print("Using CPU\n")
        device = torch.device("cpu")

    diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

    for Nepoch in config.max_epochs[2:]:
        print(f"Epoch {Nepoch}, F {config.forcing_const}\n")
        num_diff_times = 1
        PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters)
        PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
        PM = PM.to(device)

        num_paths = 10
        num_time_steps = 100
        deltaT = config.deltaT
        initial_state = np.repeat(np.array(config.initState)[np.newaxis, np.newaxis, :], num_paths, axis=0)
        assert (initial_state.shape == (num_paths, 1, config.ndims))

        true_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        global_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))
        local_states = np.zeros(shape=(num_paths, 1 + num_time_steps, config.ndims))

        # Initialise the "true paths"
        true_states[:, [0], :] = initial_state + 0.00001 * np.random.randn(*initial_state.shape)
        # Initialise the "global score-based drift paths"
        global_states[:, [0], :] = true_states[:, [0], :]
        local_states[:, [0], :] = true_states[:, [0], :]  # np.repeat(initial_state[np.newaxis, :], num_diff_times, axis=0)

        # Euler-Maruyama Scheme for Tracking Errors
        for i in tqdm(range(1, num_time_steps + 1)):
            eps = np.random.randn(num_paths, 1, config.ndims) * np.sqrt(deltaT)
            assert (eps.shape == (num_paths, 1, config.ndims))
            true_mean = true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config)

            true_states[:, [i], :] = true_states[:, [i - 1], :] \
                                     + true_drift(true_states[:, i - 1, :], num_paths=num_paths, config=config) * deltaT \
                                     + eps
            global_mean = multivar_score_based_LSTM_drift(score_model=PM, num_diff_times=num_diff_times,
                                                          diffusion=diffusion,
                                                          num_paths=num_paths, ts_step=deltaT, config=config,
                                                          device=device,
                                                          prev=global_states[:, i - 1, :])

            global_states[:, [i], :] = global_states[:, [i - 1], :] + global_mean * deltaT + eps
            local_mean = multivar_score_based_LSTM_drift(score_model=PM, num_diff_times=num_diff_times, diffusion=diffusion,
                                                         num_paths=num_paths, ts_step=deltaT, config=config, device=device,
                                                         prev=true_states[:, i - 1, :])

            local_states[:, [i], :] = true_states[:, [i - 1], :] + local_mean * deltaT + eps

        save_path = (
                project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_{config.ndims}DLorenz_DriftEvalExp_{Nepoch}Nep_tl{config.tdata_mult}data_{config.t0}t0_{config.deltaT:.3e}dT_{num_diff_times}NDT_{config.loss_factor}LFac_{config.ts_beta:.1e}Beta_{config.ts_rho:.1e}Rho_{config.ts_sigma:.1e}Sigma").replace(
            ".", "")
        print(save_path)
        np.save(save_path + "_global_true_states.npy", true_states)
        np.save(save_path + "_global_states.npy", global_states)
        np.save(save_path + "_local_states.npy", local_states)
