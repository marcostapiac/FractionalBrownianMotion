import math

import torch
import torch.nn.functional as F
from torch import nn

""" NOTE: The model below is an adaptation of the implementation of pytorch-ts """


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, diff_embed_size, diff_hidden_size, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(diff_embed_size=diff_embed_size, max_steps=max_steps),
                             persistent=False)
        self.projection1 = nn.Linear(2 * diff_embed_size, diff_hidden_size)
        self.projection2 = nn.Linear(diff_hidden_size, diff_hidden_size)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (t - low_idx).unsqueeze(-1) * (high - low)

    @staticmethod
    def _build_embedding(diff_embed_size: int, max_steps: int):
        steps = torch.arange(max_steps).unsqueeze(1)  # [max_steps,1]
        dims = torch.arange(diff_embed_size).unsqueeze(0)  # [max_steps,diff_input_size]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)  # [max_steps,diff_input_size]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
    timesteps = timesteps.view(timesteps.shape[0], )
    half_dim = embedding_dim // 2
    emb = torch.log(torch.Tensor([10000])) / (half_dim - 1)
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=torch.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]
    # noinspection PyArgumentList
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape[0] == timesteps.shape[0] and emb.shape[1] == embedding_dim
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, diffusion_hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation
        )
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1
        )
        self.diffusion_projection = nn.Linear(diffusion_hidden_size, residual_channels)  # hidden_size = 512

        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        # y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, 20, bias=False)
        self.linear2 = nn.Linear(20, int(2 * target_dim), bias=False)
        self.linear3 = nn.Linear(int(2 * target_dim), target_dim, bias=False)

    def forward(self, x):
        # CondUpSampler applies transformations on a 3D tensor of dimensions [A, B, C] on the "C" dimension, in batch
        # across [A, B]
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear3(x)
        x = F.leaky_relu(x, 0.4)
        return x

class HybridStates(nn.Module):
    def __init__(self, D, M, init_tau=1., final_tau=1.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, D))  # No fixed scaling factor
        self.b = nn.Parameter(2 * torch.pi * torch.rand(M))
        mu, sigma = math.log(10.), 2.0
        self.log_scale = nn.Parameter(torch.randn(M) * sigma + mu) # Learnable frequency magnitudes
        #self.gate_net = nn.Sequential(
        #    nn.Linear(D, D),
        #    nn.ELU(),
        #    nn.Linear(D, 2*M)
        #)
        self.gate_logits = nn.Parameter(torch.zeros(2*M))
        self.init_tau = init_tau
        self.final_tau = final_tau
        self.set_tau(self.init_tau)


    def set_tau(self, tau):
        self.tau = tau
    def forward(self, x):
        scales = torch.exp(self.log_scale).unsqueeze(1)  # [M, 1]
        W_scaled = scales * self.W                       # [M, D]
        proj = x @ W_scaled.T + self.b                   # [batch, M]
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [batch, 2M]

        # TRAIN vs EVAL for Gumbel
        if self.training:
            print(f"Used temp annealing {self.tau}\n")
            u = torch.rand_like(self.gate_logits).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            logits = self.gate_logits + gumbel
            g = torch.sigmoid(logits / self.tau).unsqueeze(0)  # [1, 2M]
        else:
            logits = self.gate_logits  # no noise at inference
            g = torch.sigmoid(logits / self.final_tau).unsqueeze(0)  # [1, 2M]
        gated_fourier = g * fourier                      # [batch, 2M]
        return gated_fourier

class MLPStateMapper(nn.Module):
    def __init__(self, ts_input_dim: int, hidden_dim: int, target_dims: int):
        super().__init__()
        M = 16
        self.hybrid = HybridStates(D=ts_input_dim, M=M)
        self.preprocess = nn.Sequential(
            nn.Linear(ts_input_dim, hidden_dim),
            nn.ELU()
        )
        self.linear2 = nn.Linear(ts_input_dim + hidden_dim + 2 * M, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, target_dims)

    def forward(self, x):
        assert (x.ndim == 3 and x.size(1) == 1) # [batch, 1, D]
        x = x.squeeze(1)    # [batch, D]

        x_raw = self.preprocess(x)            # [batch, hidden_dim]
        x_fourier = self.hybrid(x)            # [batch, 2M]
        x_combined = torch.cat([x, x_raw, x_fourier], dim=-1)  # [batch, D+hidden_dim + 2M]
        x = F.elu(self.linear2(x_combined))   # [batch, hidden_dim]
        x = self.linear3(x)                   # [batch, target_dims]
        return x.unsqueeze(1)                 # [batch, 1, target_dims]

class ConditionalMarkovianTSPostMeanScoreMatching(nn.Module):
    def __init__(
            self,
            max_diff_steps: int,
            diff_embed_size: int,
            diff_hidden_size: int,
            ts_dims: int,
            mlp_hidden_dims:int,
            condupsampler_length:int = 20,
            residual_layers: int = 10,
            residual_channels: int = 8,
            dilation_cycle_length: int = 10
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1
        )
        self.diffusion_embedding = DiffusionEmbedding(diff_embed_size=diff_embed_size,
                                                      diff_hidden_size=diff_hidden_size,
                                                      max_steps=max_diff_steps)  # get_timestep_embedding

        self.mlp_state_mapper = MLPStateMapper(ts_input_dim=ts_dims, hidden_dim=mlp_hidden_dims, target_dims=condupsampler_length)
        self.cond_upsampler = CondUpsampler(
            target_dim=ts_dims, cond_length=condupsampler_length
        )#target_dim = 1 or target_dim = ts_dims
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    diffusion_hidden_size=diff_hidden_size
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = nn.Conv1d(residual_channels, 1, 1)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, times, conditioner, eff_times):
        # inputs = inputs.unsqueeze(1)
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.01)

        diffusion_step = self.diffusion_embedding(times)
        conditioner = self.mlp_state_mapper(conditioner)
        cond_up = self.cond_upsampler(conditioner)
        skip = []
        for layer in self.residual_layers:
            if cond_up.shape[1] != 1:
                cond_up = cond_up.reshape(cond_up.shape[0]*cond_up.shape[1], 1, -1)
            x, skip_connection = layer(x, conditioner=cond_up, diffusion_step=diffusion_step)
            x = F.leaky_relu(x, 0.01)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.01)
        x = self.output_projection(x)
        # For VPSDE only
        beta_tau = torch.exp(-0.5 * eff_times)
        sigma2_tau = (1. - torch.exp(-eff_times))
        # Network tries to learn the posterior mean
        return -inputs / sigma2_tau + (beta_tau / sigma2_tau) * x
