from typing import Callable

import torch
from torch import nn
from torch.functional import F


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

    def _build_embedding(self, diff_embed_size: int, max_steps: int):
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
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape[0] == timesteps.shape[0] and emb.shape[1] == embedding_dim
    return emb


class MLP(nn.Module):
    def __init__(self, input_shape: int, hidden_shapes: list, output_shape, bias: bool = True, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        self.bias = bias
        i = 0
        self.hiddenLayers = nn.ModuleList()
        while i < len(self.hidden_shapes):
            if i == 0:
                newLayer = nn.Sequential(
                    nn.Linear(in_features=self.input_shape, out_features=self.hidden_shapes[i], bias=self.bias))
            else:
                newLayer = nn.Sequential(
                    nn.Linear(in_features=self.hidden_shapes[i - 1], out_features=self.hidden_shapes[i],
                              bias=self.bias))
            self.hiddenLayers.append(newLayer)
            i += 1
        self.finalLayer = nn.Linear(in_features=self.hidden_shapes[-1], out_features=self.output_shape,
                                    bias=False)  # x.shape[-1]

    def forward(self, x: torch.Tensor, act: Callable, *args):
        for layer in self.hiddenLayers:
            x = layer(x)
            x = act(x, *args)
        x = self.finalLayer.forward(x)
        return x


class NaiveMLP(nn.Module):
    """Create a naive MLP network.

    Args:
        output_shape (int): output shape.
        enc_shapes (int): The shapes of the encoder.
        t_dim (int): the dimension of the time embedding.
        dec_shapes (int): The shapes of the decoder
        resnet (bool): if True then the network is a resnet.
    """

    def __init__(
            self,
            temb_dim: int,
            max_diff_steps: int,
            output_shape: int,
            enc_shapes: list,
            dec_shapes: list,
    ):
        super().__init__()
        self.temb_dim = temb_dim
        t_enc_dim = temb_dim * 2
        self.diffusion_embedding = DiffusionEmbedding(diff_embed_size=self.temb_dim, diff_hidden_size=self.temb_dim,
                                                      max_steps=max_diff_steps)  # get_timestep_embedding

        self.t_encoder = MLP(input_shape=self.temb_dim,
                             hidden_shapes=enc_shapes, output_shape=t_enc_dim
                             )

        self.x_encoder = MLP(input_shape=output_shape,
                             hidden_shapes=enc_shapes, output_shape=t_enc_dim
                             )

        self.net = MLP(input_shape=t_enc_dim * 2, hidden_shapes=dec_shapes,
                       output_shape=output_shape)

    def forward(self, inputs: torch.Tensor, times: torch.Tensor):
        if len(inputs.shape) == 3:
            x = inputs.squeeze(1)
        assert (len(x.shape) == 2 and len(times.shape) == 1 and x.shape[0] == times.shape[0])
        temb = self.diffusion_embedding(times.reshape(-1))  # , self.temb_dim)
        assert (temb.shape[0] == times.shape[0] and temb.shape[1] == self.temb_dim)
        temb = self.t_encoder.forward(temb, F.leaky_relu, 0.4)
        assert (temb.shape[0] == times.shape[0] and temb.shape[1] == self.temb_dim * 2)
        xemb = self.x_encoder(x, F.leaky_relu, 0.4)
        assert (xemb.shape[0] == x.shape[0] and xemb.shape[1] == self.temb_dim * 2)
        temb = torch.broadcast_to(temb, [xemb.shape[0], *temb.shape[1:]])
        h = torch.cat([xemb, temb], dim=-1)
        assert (h.shape[0] == x.shape[0] and h.shape[1] == self.temb_dim * 2 * 2)
        out = -self.net.forward(h, F.leaky_relu, 0.4)
        return out


"""
def init_model(key, batch_size=256, dimension=2):
    def forward(t, x):
        model = Naive(output_shape=dimension,
                      enc_shapes=[32, 32],
                      t_dim=16,
                      dec_shapes=[32, 32])
        return model(t, x)

    model = torch.transform(forward)

    input_shape = (batch_size, dimension)
    t_shape = (batch_size, 1)
    dummy_t = torch.zeros(t_shape)
    dummy_input = torch.zeros(input_shape)

    init_params = model.init(key, t=dummy_t, x=dummy_input)
    return model, init_params
"""
