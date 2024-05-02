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
        # Note a linear layer expects the input vector to have in_features in the last dimension
        self.linear1 = nn.Linear(in_features=cond_length, out_features=int(2 * target_dim), bias=False)
        self.linear2 = nn.Linear(in_features=int(2 * target_dim), out_features=target_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x


class ConditionalLSTMTSSampleScoreMatching(nn.Module):
    def __init__(
            self,
            max_diff_steps: int,
            diff_embed_size: int,
            diff_hidden_size: int,
            lstm_hiddendim: int,
            lstm_numlay: int,
            lstm_inputdim: int = 1,
            lstm_dropout: float = 0.1,
            residual_layers: int = 10,
            residual_channels: int = 8,
            dilation_cycle_length: int = 10
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=lstm_inputdim,  # What is the input_size of an LSTM?
            hidden_size=lstm_hiddendim,
            num_layers=lstm_numlay,
            dropout=lstm_dropout,
            batch_first=True,
        )
        # For input of size (B, T, D), input projection applies cross-correlation for each t along D dimensions
        # So if we have processed our B time-series of length T and dimension D into (BT, 1, D) then input projection
        # accumulates spatial information mapping each (1, D) tensor into a (residual_channel, Lout) tensor
        # where Lout is a function of D and convolution parameters
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1
        )
        self.diffusion_embedding = DiffusionEmbedding(diff_embed_size=diff_embed_size,
                                                      diff_hidden_size=diff_hidden_size,
                                                      max_steps=max_diff_steps)  # get_timestep_embedding

        # For input of shape (B, 1, N)
        # Target dim is the dimension of output vector
        # Cond_length is the dimension of the input vector (N)
        # As a linear layer, it expects input to be a vector, not a matrix
        self.cond_upsampler = CondUpsampler(
            target_dim=1, cond_length=lstm_hiddendim
        )
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

    def forward(self, inputs, times, conditioner, beta_tau, sigma_tau):
        # For Conditional Time series, input projection accumulates information spatially
        # Therefore it expects inputs to be of shape (BatchSize, 1, NumDims)
        if torch.any(torch.isnan(inputs)):
            print(f"0:{torch.any(torch.isnan(inputs))}\n")
            raise RuntimeError
        x = self.input_projection(inputs)
        if torch.any(torch.isnan(x)):
            print(f"1:{torch.any(torch.isnan(x))}\n")
            raise RuntimeError
        x = F.leaky_relu(x, 0.01)
        if torch.any(torch.isnan(x)):
            print(f"2:{torch.any(torch.isnan(x))}\n")
            raise RuntimeError
        diffusion_step = self.diffusion_embedding(times)
        if torch.any(torch.isnan(diffusion_step)):
            print(f"2:{torch.any(torch.isnan(diffusion_step))}\n")
            raise RuntimeError
        # Linear layer assumes dimension of conditioning vector to be in last dimension
        # This conditioner needs to be of shape (BatchSize, 1, NumFeatDims)
        cond_up = self.cond_upsampler(conditioner)
        print(f"4:{torch.any(torch.isnan(cond_up))}\n")
        skip = []
        i = 0
        for layer in self.residual_layers:
            x, skip_connection = layer(x, conditioner=cond_up, diffusion_step=diffusion_step)
            print(f"5_{i}:{torch.any(torch.isnan(x))}\n")
            x = F.leaky_relu(x, 0.01)
            print(f"6_{i}:{torch.any(torch.isnan(x))}\n")
            skip.append(skip_connection)
            i+=1

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        print(f"7:{torch.any(torch.isnan(x))}\n")
        x = self.skip_projection(x)
        print(f"8:{torch.any(torch.isnan(x))}\n")
        x = F.leaky_relu(x, 0.01)
        print(f"9:{torch.any(torch.isnan(x))}\n")
        x = self.output_projection(x)
        print(f"10:{torch.any(torch.isnan(x))}\n")
        assert (inputs.shape == x.shape == beta_tau.shape == sigma_tau.shape)
        print(torch.any(torch.isnan(x)))
        return -torch.pow(sigma_tau, -1) * (inputs - beta_tau * x)
