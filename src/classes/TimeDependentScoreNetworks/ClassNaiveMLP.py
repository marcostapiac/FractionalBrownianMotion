import numpy as np
import torch
from torch import nn


def get_timestep_embedding(timesteps: torch.Tensor,
                           embedding_dim: int,
                           max_positions=10000) -> torch.Tensor:
    """ Get timesteps embedding.
    Function extracted from https: // github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Args:
        timesteps (jnp.ndarray): timesteps array (Nbatch,).
        embedding_dim (int): Size of the embedding.
        max_positions (int, optional): _description_. Defaults to 10000.

    Returns:
        emb (jnp.ndarray): embedded timesteps (Nbatch, embedding_dim).
    """
    assert len(timesteps.shape) == 1
    assert (embedding_dim % 2 == 0)
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = torch.log(torch.from_numpy(np.array([max_positions]))) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.concatenate([torch.sin(emb), torch.cos(emb)], axis=1)
    # if embedding_dim % 2 == 1:  # zero pad
    #    emb = F.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class MLP(nn.Module):
    def __init__(self, input_shape: int, hidden_shapes: list, output_shape, bias: bool = True, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.hidden_shapes = hidden_shapes
        self.output_shape = output_shape
        self.bias = bias
        i = 0
        self.hiddenLayers = []
        while i < len(self.hidden_shapes):
            hs = self.hidden_shapes[i]
            newLayer = nn.Sequential(
                nn.Linear(in_features=self.input_shape, out_features=hs, bias=self.bias)
            ) if i == 0 else nn.Sequential(
                nn.Linear(in_features=hs, out_features=hs, bias=self.bias)
            )
            self.hiddenLayers.append(newLayer)
            i += 1
        self.finalLayer = nn.Linear(in_features=self.hidden_shapes[-1], out_features=self.output_shape,
                                    bias=False)  # x.shape[-1]

    def forward(self, x):
        for layer in self.hiddenLayers:
            x = layer(x)
            x = torch.sin(x)
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
            output_shape: int,
            enc_shapes: list,
            temb_dim: int,
            dec_shapes: list,
    ):
        super().__init__()
        self.temb_dim = temb_dim
        t_enc_dim = temb_dim * 2

        self.net = MLP(input_shape=t_enc_dim * 2, hidden_shapes=dec_shapes,
                       output_shape=output_shape)

        self.t_encoder = MLP(input_shape=self.temb_dim,
                             hidden_shapes=enc_shapes, output_shape=t_enc_dim
                             )

        self.x_encoder = MLP(input_shape=output_shape,
                             hidden_shapes=enc_shapes, output_shape=t_enc_dim
                             )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # t = torch.from_numpy(np.array([t])).to(torch.float32)
        if len(x.shape) == 3:
            x = x.squeeze(1)
        assert (len(x.shape) > 1 and len(t.shape) > 1 and x.shape[0] == t.shape[0])
        temb = get_timestep_embedding(t.reshape(-1), self.temb_dim)
        assert (temb.shape[0] == t.shape[0] and temb.shape[1] == self.temb_dim)
        temb = self.t_encoder(temb)
        assert (temb.shape[0] == t.shape[0] and temb.shape[1] == self.temb_dim * 2)
        xemb = self.x_encoder(x)
        assert (xemb.shape[0] == x.shape[0] and xemb.shape[1] == self.temb_dim * 2)
        temb = torch.broadcast_to(temb, [xemb.shape[0], *temb.shape[1:]])
        h = torch.cat([xemb, temb], dim=-1)
        assert (h.shape[0] == x.shape[0] and h.shape[1] == self.temb_dim * 2 * 2)
        out = -self.net(h)
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
