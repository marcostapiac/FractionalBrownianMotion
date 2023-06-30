import math

import torch
import torch.nn.functional as F
from torch import nn

""" NOTE: The model below is an adaptation of the implementation of pytorch-ts """

"""
class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=10000):  # Changed maximum number of diffusion steps
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table
"""


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
    assert len(timesteps.shape) == 2 and timesteps.shape[1] == 1  # and timesteps.dtype == tf.int32
    timesteps = timesteps.view(timesteps.shape[0], )
    half_dim = embedding_dim // 2
    emb = torch.log(torch.Tensor([10000])) / (half_dim - 1)
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=torch.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape[0] == timesteps.shape[0] and emb.shape[1] == embedding_dim
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1, padding=2, padding_mode="circular"
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)

        y = x + diffusion_step
        y = self.dilated_conv(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class TimeSeriesNoiseMatching(nn.Module):
    def __init__(
            self,
            time_emb_dim=128,
            residual_layers=8,
            residual_channels=8,
            dilation_cycle_length=2,
            residual_hidden=64,
    ):
        super().__init__()
        self.time_emb_dim = residual_hidden
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1, padding=2  # ,padding_mode="circular"
        )
        self.diffusion_embedding = get_timestep_embedding  # DiffusionEmbedding(
        # time_emb_dim, proj_dim=residual_hidden
        # )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, times):
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)
        diffusion_step = self.diffusion_embedding(times, embedding_dim=self.time_emb_dim)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step)
            x = F.leaky_relu(x, 0.4)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x
