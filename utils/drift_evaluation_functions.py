import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def multivar_score_based_LSTM_drift(score_model, num_diff_times, diffusion, num_paths, prev, ts_step, config,
                                    device):
    """ Computes drift using LSTM score network when features obtained from in-sample data """
    score_model = score_model.to(device)
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    if prev[0, :].shape[0] > 1:
        features = find_LSTM_feature_vectors_multiDTS(Xs=prev, score_model=score_model, device=device, config=config)
        num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in prev}
    else:
        features = find_LSTM_feature_vectors_oneDTS(Xs=prev, score_model=score_model, device=device, config=config)
        num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in prev}
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


def multivar_score_based_LSTM_drift_OOS(score_model, time_idx, h, c, num_diff_times, diffusion, num_paths, prev,
                                        ts_step, config,
                                        device):
    """ Computes drift using LSTM score network when features obtained from LSTM directly """

    score_model = score_model.to(device)
    num_taus = 100
    Ndiff_discretisation = config.max_diff_steps
    assert (prev.shape == (num_paths, config.ndims))
    score_model.eval()
    with torch.no_grad():
        if time_idx == 0:
            features, (h, c) = score_model.rnn(torch.tensor(prev[:, np.newaxis, :], dtype=torch.float32).to(device),
                                               None)
        else:
            features, (h, c) = score_model.rnn(torch.tensor(prev[:, np.newaxis, :], dtype=torch.float32).to(device),
                                               (h, c))
    assert (prev[0, :].shape[0] == config.ts_dims)
    if prev[0, :].shape[0] > 1:
        features = {tuple(prev[i, :].squeeze().tolist()): features[i] for i in range(prev.shape[0])}
        num_feats_per_x = {tuple(x.squeeze().tolist()): features[tuple(x.squeeze().tolist())].shape[0] for x in prev}
    else:
        features = {prev[i, :].item(): features[i] for i in range(prev.shape[0])}
        num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in prev}
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
    return means.mean(dim=0).reshape(num_paths, 1, config.ts_dims).cpu().numpy(), h, c


def find_LSTM_feature_vectors_multiDTS(Xs, score_model, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float).to(device)

    def process_single_threshold(x, score_model, device, dX_global):
        diff = sim_data_tensor - x.reshape(1, -1)  # shape: (M, N, D)
        diff = diff.norm(dim=-1)  # Result: (M, N)
        mask = diff <= torch.min(diff)

        """mask = diff <=np.arccos(dX_global)
        thresh = dX_global
        while torch.sum(mask) == 0:
            thresh = np.cos(np.arccos(thresh)*2)
            mask = diff <= np.arccos(thresh)
        assert (torch.sum(mask) > 0)"""

        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)
        sequences = []
        js = []
        for k in range(min(100, indices.shape[0])):
            idx = indices[k, :]
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1, :]
            sequences.append(seq)
            js.append(len(seq))
        outputs = []
        score_model = score_model.to(device)
        score_model.eval()
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan).to(device)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            # padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = score_model.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    features_Xs = {}
    dX_global = np.cos(1. / 5000)
    for i in range(Xs.shape[0]):
        x = Xs[i, :].reshape(-1, 1)
        x_val, out = process_single_threshold(torch.tensor(x).to(device, dtype=torch.float), device=device,
                                              score_model=score_model, dX_global=dX_global)
        assert (len(out) > 0)
        features_Xs[tuple(x.squeeze().tolist())] = out
    return features_Xs


def find_LSTM_feature_vectors_oneDTS(Xs, score_model, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float).to(device)

    def process_single_threshold(x, score_model, device, dX):
        # Compute the mask over the entire sim_data matrix
        diff = sim_data_tensor - x
        mask = torch.abs(diff) <= torch.min(torch.abs(diff))
        """
        xmin = x - dX
        xmax = x + dX
        mask = (sim_data_tensor >= xmin) & (sim_data_tensor <= xmax)
        while torch.sum(mask) == 0:
            dX *= 2
            xmin = x - dX
            xmax = x + dX
            # Compute the mask over the entire sim_data matrix
            mask = (sim_data_tensor >= xmin) & (sim_data_tensor <= xmax)
        """
        assert torch.sum(mask) > 0
        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)

        sequences = []
        js = []
        for k in range(min(100, indices.shape[0])):
            idx = indices[k, :]
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1]
            sequences.append(seq)
            js.append(len(seq))

        outputs = []
        score_model = score_model.to(device)
        score_model.eval()
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = score_model.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    # Option 1: Process sequentially (using tqdm)
    features_Xs = {}
    assert (len(Xs.shape) == 1 or Xs.shape[-1] == 1)
    if len(Xs.shape) == 1:  # Domain RMSE
        dX_global = np.diff(Xs)[0] / 5000
        assert (((Xs[1] - Xs[0]) / 5000) == dX_global)
        for x in (Xs):
            x_val, out = process_single_threshold(x, device=device, score_model=score_model, dX=dX_global)
            assert (len(out) > 0)
            features_Xs[x_val.item()] = out
    else:
        for i in range(Xs.shape[0]):
            dX_global = 1. / 5000
            x = Xs[i, :].reshape(-1, 1)
            assert (x.shape[-1] == 1 and x.shape[0] == 1)
            x_val, out = process_single_threshold(x[0, 0], score_model=score_model, device=device, dX=dX_global)
            assert (len(out) > 0)
            features_Xs[x_val.item()] = out
    return features_Xs


def multivar_gaussian_kernel(inv_H, norm_const, x):
    """exponent = -0.5 * np.einsum('...i,ij,...j', x, inv_H, x)
    print(exponent.shape, x.shape)
    return norm_const * np.exp(exponent)"""
    if torch.cuda.is_available():
        device = 0
    else:
        device = torch.device("cpu")
    x = torch.tensor(x, dtype=torch.float32, device=device)
    inv_H = torch.tensor(inv_H, dtype=torch.float32, device=device)
    y = torch.matmul(x, inv_H)  # shape: (N, T1, T2, D)
    # Compute the dot product along the last dimension.
    # This is equivalent to the einsum: '...i, ...i'
    exponent = -0.5 * torch.sum(x * y, dim=-1)  # shape: (N, T1, T2)
    # Return the computed Gaussian kernel.
    return (norm_const * torch.exp(exponent)).cpu().numpy()


def IID_NW_multivar_estimator(prevPath_observations, path_incs, inv_H, norm_const, x, t1, t0, truncate):
    if len(prevPath_observations.shape) > 2:
        N, n, d = prevPath_observations.shape
        kernel_weights_unnorm = multivar_gaussian_kernel(inv_H=inv_H, norm_const=norm_const,
                                                         x=prevPath_observations[:, :, np.newaxis, :] - x[np.newaxis,
                                                                                                        np.newaxis,
                                                                                                        :, :])
        denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
        assert (denominator.shape == (x.shape[0], 1))
        numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, :], axis=(1, 0)) / N * (
                t1 - t0)
    elif len(prevPath_observations.shape) == 2:
        N, n = prevPath_observations.shape
        kernel_weights_unnorm = multivar_gaussian_kernel(inv_H=inv_H, norm_const=norm_const,
                                                         x=prevPath_observations[:, :, np.newaxis, np.newaxis] - x[
                                                                                                                 np.newaxis,
                                                                                                                 np.newaxis,
                                                                                                                 :, :])
        denominator = np.sum(kernel_weights_unnorm, axis=(1, 0))[:, np.newaxis] / (N * n)
        assert (denominator.shape == (x.shape[0], 1))
        numerator = np.sum(kernel_weights_unnorm[..., np.newaxis] * path_incs[:, :, np.newaxis, np.newaxis],
                           axis=(1, 0)) / N * (
                            t1 - t0)
    assert (numerator.shape == x.shape)
    estimator = numerator / denominator
    assert (estimator.shape == x.shape)
    # assert all([np.all(estimator[i, :] == numerator[i,:]/denominator[i,0]) for i in range(estimator.shape[0])])
    # This is the "truncated" discrete drift estimator to ensure appropriate risk bounds
    if truncate:
        m = np.min(denominator[:, 0])
        estimator[denominator[:, 0] <= m / 2., :] = 0.
    return estimator
