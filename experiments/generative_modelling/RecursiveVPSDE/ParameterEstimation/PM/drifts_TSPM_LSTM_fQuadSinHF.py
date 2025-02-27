import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from configs import project_config
from configs.RecursiveVPSDE.LSTM_fQuadSinHF.recursive_LSTM_PostMeanScore_fQuadSinHF_T256_H05_tl_110data import \
    get_config as get_config
from src.generative_modelling.models.ClassVPSDEDiffusion import VPSDEDiffusion
from src.generative_modelling.models.TimeDependentScoreNetworks.ClassConditionalLSTMTSPostMeanScoreMatching import \
    ConditionalLSTMTSPostMeanScoreMatching
from tqdm import tqdm


# In[22]:

def find_LSTM_feature_vectors(Xs, PM, config, device):
    sim_data = np.load(config.data_path, allow_pickle=True)
    sim_data_tensor = torch.tensor(sim_data, dtype=torch.float)
    dX = np.diff(Xs)[0] / 250
    assert (((Xs[1] - Xs[0]) / 250) ==  dX)

    def process_single_threshold(x):
        xmin = x - dX
        xmax = x + dX
        # Compute the mask over the entire sim_data matrix
        mask = (sim_data_tensor >= xmin) & (sim_data_tensor <= xmax)
        # Get indices where mask is True (each index is [i, j])
        indices = mask.nonzero(as_tuple=False)

        sequences = []
        js = []
        for idx in indices:
            i, j = idx.tolist()
            # Extract the sequence: row i, columns 0 to j (inclusive)
            seq = sim_data_tensor[i, :j + 1]
            sequences.append(seq)
            js.append(len(seq))

        outputs = []
        if sequences:
            # Pad sequences to create a batch.
            # pad_sequence returns tensor of shape (batch_size, max_seq_len)
            padded_batch = pad_sequence(sequences, batch_first=True, padding_value=torch.nan)
            # Add feature dimension: now shape becomes (batch_size, max_seq_len, 1)
            padded_batch = padded_batch.unsqueeze(-1).to(device)
            with torch.no_grad():
                batch_output, _ = PM.rnn(padded_batch, None)
            outputs = batch_output[torch.arange(batch_output.shape[0]), torch.tensor(js, dtype=torch.long) - 1,
                      :].unsqueeze(1).cpu()
        return x, outputs

    # Option 1: Process sequentially (using tqdm)
    features_Xs = {}
    for x in (Xs):
        x_val, out = process_single_threshold(x)
        assert (len(out) > 0)
        features_Xs[x_val.item()] = out

    return features_Xs


config = get_config()

print("Beta Min : ", config.beta_min)
if config.has_cuda:
    device = int(os.environ["LOCAL_RANK"])
else:
    print("Using CPU\n")
    device = torch.device("cpu")

diffusion = VPSDEDiffusion(beta_max=config.beta_max, beta_min=config.beta_min)

max_diff_steps = config.max_diff_steps
sample_eps = config.sample_eps
ts_step = config.deltaT

Nepoch = 300
# Fix the number of training epochs and training loss objective loss
PM = ConditionalLSTMTSPostMeanScoreMatching(*config.model_parameters).to(device)
PM.load_state_dict(torch.load(config.scoreNet_trained_path + "_NEp" + str(Nepoch)))
print(config.scoreNet_trained_path)

Xshape = config.ts_length
num_taus = 200

num_diff_times = config.max_diff_steps
Ndiff_discretisation = config.max_diff_steps
diffusion_times = torch.linspace(start=config.sample_eps, end=config.end_diff_time,
                                 steps=Ndiff_discretisation).to(device)

Xs = torch.linspace(-1.2, 1.2, steps=Xshape)
features = find_LSTM_feature_vectors(Xs=Xs, PM=PM, device=device, config=config)
num_feats_per_x = {x.item(): features[x.item()].shape[0] for x in Xs}
list_num_feats_per_x = list(num_feats_per_x.values())
tot_num_feats = np.sum(list(num_feats_per_x.values()))
features_tensor = torch.concat(list(features.values()), dim=0).to(device)  # [num_features_per_x, 1, 20]
assert (features_tensor.shape[0] == tot_num_feats)
final_vec_mu_hats = np.zeros((Xshape, num_diff_times, num_taus, config.ts_dims))  # Xvalues, DiffTimes, Ztaus, Ts_Dims

vec_Z_taus = diffusion.prior_sampling(shape=(tot_num_feats * num_taus, 1, config.ts_dims)).to(device)
ts = []
es = 1
mu_hats_mean = np.zeros((tot_num_feats, num_taus))
mu_hats_std = np.zeros((tot_num_feats, num_taus))
difftime_idx = num_diff_times - 1

PM.eval()
while difftime_idx >= num_diff_times - es:
    d = diffusion_times[Ndiff_discretisation - (num_diff_times - 1 - difftime_idx) - 1].to(device)
    diff_times = torch.stack([d for _ in range(tot_num_feats)]).reshape(tot_num_feats * 1).to(device)
    eff_times = diffusion.get_eff_times(diff_times=diff_times).unsqueeze(-1).unsqueeze(-1).to(device)
    vec_diff_times = torch.stack([diff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats)
    vec_eff_times = torch.stack([eff_times for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats, 1, 1)
    vec_conditioner = torch.stack([features_tensor for _ in range(num_taus)], dim=0).reshape(num_taus * tot_num_feats,
                                                                                             1, -1)

    # assert (all([torch.allclose(vec_conditioner[j::tot_num_feats,:,:],vec_conditioner[j,:,:]) for j in range(Xshape)]))
    # vec_c_mat = vec_conditioner.reshape((num_taus, tot_num_feats, vec_conditioner.shape[-1])).cpu()
    # for j in range(Xshape):
    #    c = vec_c_mat[:, sum(list_num_feats_per_x[:j]):sum(list_num_feats_per_x[:j+1]), :]
    #    assert torch.allclose(c, c[0, :, :])
    #    assert torch.allclose(c[0, :, :], features[Xs[j].item()].squeeze(1))
    # del vec_c_mat
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

type = "PM"
assert (type in config.scoreNet_trained_path)
print(type)

save_path = (
        project_config.ROOT_DIR + f"experiments/results/TSPM_LSTM_fQuadSinHF2_DriftEvalExp_{Nepoch}Nep_{config.loss_factor}LFactor_{config.t0}t0_{config.deltaT:.3e}dT_{config.quad_coeff}a_{config.sin_coeff}b_{config.sin_space_scale}c_{config.max_diff_steps}DiffSteps").replace(
    ".", "")
print(save_path)
assert (final_vec_mu_hats.shape == (Xshape, num_diff_times, num_taus, config.ts_dims))
assert config.ts_dims == 1
np.save(save_path + "_muhats.npy", final_vec_mu_hats[:, -es:, :, 0])
