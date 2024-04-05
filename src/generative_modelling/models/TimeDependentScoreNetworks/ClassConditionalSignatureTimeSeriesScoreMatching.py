import math
import signatory
import torch
import torch.nn.functional as F
from torch import nn

from utils.math_functions import time_aug

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


class SigNet(nn.Module):
    def __init__(self, in_dims: int,  out_dims:int, sig_depth: int):
        super(SigNet, self).__init__()
        self.augment = time_aug
        self.conv1d = torch.nn.Conv1d(in_channels=in_dims + 1, out_channels=in_dims + 1, padding=0, kernel_size=1,
                                      stride=1)
        self.signature = signatory.Signature(depth=sig_depth, stream=True)
        self.linear = torch.nn.Linear(in_features=out_dims, out_features=out_dims)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        # Batch is of shape (N, T, D)
        N, T, _ = batch.shape
        a = self.augment(batch, time_ax= torch.atleast_2d((torch.arange(1, T + 1) / T)).T.to(batch.device))
        # Batch is of shape (N, T, D+1)
        b = self.conv1d(a.permute(0, 2, 1)).permute((0,2,1))
        # Batch is now of shape (N, T, D+1)
        c = self.signature(b, basepoint=True)
        # Signatures are now of shape (N, T, NSIGFEATS)
        c = torch.concat([torch.zeros(size=(c.shape[0], c.shape[-1])), c[:, 1:, :]], dim=1)
        print(batch[0,:,:], a[0,:,:], c[0,:,:])
        # Features are delayed path signatures
        # Now pass each feature through a simple feedforward network
        d = self.linear(c)
        d = torch.nn.functional.tanh(d)
        return d


class ConditionalSignatureTimeSeriesScoreMatching(nn.Module):
    def __init__(
            self,
            max_diff_steps: int,
            diff_embed_size: int,
            diff_hidden_size: int,
            ts_dims: int,
            sig_depth:int,
            feat_hiddendims:int,
            residual_layers: int = 10,
            residual_channels: int = 8,
            dilation_cycle_length: int = 10
    ):
        super().__init__()
        # For input of size (B, T, D), input projection applies cross-correlation for each t along D dimensions
        # So if we have processed our B time-series of length T and dimension D into (BT, 1, D) then input projection
        # accumulates spatial information mapping each (1, D) tensor into a (residual_channel, Lout) tensor
        # where Lout is a function of D and convolution parameters
        self.signet = SigNet(in_dims=ts_dims, out_dims=feat_hiddendims,sig_depth=sig_depth)
        self.input_projection = nn.Conv1d(
            in_channels=ts_dims, out_channels=residual_channels, kernel_size=1
        )
        self.diffusion_embedding = DiffusionEmbedding(diff_embed_size=diff_embed_size,
                                                      diff_hidden_size=diff_hidden_size,
                                                      max_steps=max_diff_steps)  # get_timestep_embedding

        # For feature of shape (B, 1, N)
        # Target dim is the dimension of output vector
        # Cond_length is the dimension of the feature vector (N)
        # As a linear layer, it expects input to be a vector, not a matrix
        self.cond_upsampler = CondUpsampler(
            target_dim=1, cond_length=feat_hiddendims
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

    def forward(self, inputs, times, conditioner):
        # inputs = inputs.unsqueeze(1)
        # For Conditional Time series, input projection accumulates information spatially
        # Therefore it expects inputs to be of shape (BatchSize, 1, NumDims)
        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.01)

        diffusion_step = self.diffusion_embedding(times)
        # Linear layer assumes dimension of conditioning vector to be in last dimension
        # This conditioner needs to be of shape (BatchSize, 1, NumFeatDims)
        cond_up = self.cond_upsampler(conditioner)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, conditioner=cond_up, diffusion_step=diffusion_step)
            x = F.leaky_relu(x, 0.01)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.01)
        x = self.output_projection(x)
        return x
