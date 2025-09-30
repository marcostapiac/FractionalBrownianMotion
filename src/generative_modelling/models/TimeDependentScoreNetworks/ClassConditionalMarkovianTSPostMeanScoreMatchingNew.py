import math
import torch
import torch.nn.functional as F
from torch import nn

@torch.jit.script
def silu(x):  # keep for DiffusionEmbedding
    return x * torch.sigmoid(x)

class DiffusionEmbedding(nn.Module):
    def __init__(self, diff_embed_size, diff_hidden_size, max_steps):
        super().__init__()
        self.register_buffer(
            'embedding',
            self._build_embedding(diff_embed_size=diff_embed_size, max_steps=max_steps),
            persistent=False
        )
        self.projection1 = nn.Linear(2 * diff_embed_size, diff_hidden_size)
        self.projection2 = nn.Linear(diff_hidden_size, diff_hidden_size)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x); x = silu(x)
        x = self.projection2(x); x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx  = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low, high = self.embedding[low_idx], self.embedding[high_idx]
        return low + (t - low_idx).unsqueeze(-1) * (high - low)

    @staticmethod
    def _build_embedding(diff_embed_size: int, max_steps: int):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims  = torch.arange(diff_embed_size).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    timesteps = timesteps.view(timesteps.shape[0], )
    half_dim = embedding_dim // 2
    emb = torch.log(torch.Tensor([10000])) / (half_dim - 1)
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:
        emb = torch.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape[0] == timesteps.shape[0] and emb.shape[1] == embedding_dim
    return emb

# -------- Residual Block (channels only) --------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dilated_conv = nn.Conv1d(channels, 2*channels, kernel_size=1, padding=0) # This was changed

        self.conditioner_projection = nn.Conv1d(1, 2 * channels, kernel_size=1)
        self.output_projection      = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, time_bias):  # x:[B,C,D], conditioner:[B,1,D], time_bias:[B,C,1]
        # add per-channel time bias (broadcast across D)
        y = x + time_bias
        y = self.dilated_conv(y) + self.conditioner_projection(conditioner)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, 20, bias=False)
        self.linear2 = nn.Linear(20, int(2 * target_dim), bias=False)
        self.linear3 = nn.Linear(int(2 * target_dim), target_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x); x = F.silu(x)
        x = self.linear2(x); x = F.silu(x)
        x = self.linear3(x); x = F.silu(x)
        return x

class HybridStates(nn.Module):
    def __init__(self, D, M, init_tau=1., final_tau=1.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, D))
        self.b = nn.Parameter(2 * torch.pi * torch.rand(M))
        mu, sigma = math.log(10.), 2.0
        self.log_scale = nn.Parameter(torch.randn(M) * sigma + mu)
        self.gate_mlp  = nn.Sequential(nn.Linear(D, 32), nn.ELU(), nn.Linear(32, 2*M))
        self.init_tau, self.final_tau = init_tau, final_tau
        self.tau = init_tau

    def set_tau(self, tau): self.tau = tau

    def forward(self, x):
        scales   = torch.exp(self.log_scale).unsqueeze(1)  # [M,1]
        W_scaled = scales * self.W                         # [M,D]
        proj     = x @ W_scaled.T + self.b                 # [B,M]
        fourier  = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B,2M]
        temp  = self.tau if self.training else self.final_tau
        alpha = 0.1
        g = 1.0 + alpha * torch.tanh(self.gate_mlp(x) / temp)             # [B,2M]
        return g * fourier

class TokenISAB(nn.Module):
    """
    Induced Self-Attention Block over tokens (dims):
      H = MHA(Z, X, X)       # Z: m learnable inducing tokens
      Y = MHA(X, H, H)
    Shapes: X:[B,C,D] -> [B,C,D]
    """
    def __init__(self, C: int, m: int = 64, heads: int = 4, p: float = 0.0):
        super().__init__()
        self.m = m
        self.Z = nn.Parameter(torch.randn(m, C))
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.mha1 = nn.MultiheadAttention(embed_dim=C, num_heads=heads, batch_first=True, dropout=p)
        self.mha2 = nn.MultiheadAttention(embed_dim=C, num_heads=heads, batch_first=True, dropout=p)
        self.ffn  = nn.Sequential(nn.Linear(C, 4*C), nn.GELU(), nn.Linear(4*C, C))

    def forward(self, x):                      # x: [B,C,D]
        y = x.transpose(1, 2)                  # [B,D,C]
        z = self.Z.unsqueeze(0).expand(y.size(0), -1, -1)              # [B,m,C]
        h, _ = self.mha1(self.ln1(z), y, y, need_weights=False)        # [B,m,C]
        y2, _ = self.mha2(self.ln2(y), h, h, need_weights=False)       # [B,D,C]
        y = y + y2                                                     # resid
        y = y + self.ffn(y)                                            # MLP resid
        return y.transpose(1, 2)
class MLPStateMapper(nn.Module):
    def __init__(self, ts_input_dim: int, hidden_dim: int, target_dims: int):
        super().__init__()
        M = 16
        self.hybrid = HybridStates(D=ts_input_dim, M=M)
        self.preprocess = nn.Sequential(nn.Linear(ts_input_dim, hidden_dim), nn.ELU())
        self.linear2 = nn.Linear(4*ts_input_dim + hidden_dim + 2 * M, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, target_dims)

    def forward(self, x):
        assert (x.ndim == 3 and x.size(1) == 1)  # [B,1,D]
        x = x.squeeze(1)                          # [B,D]
        x_poly = torch.cat([torch.pow(x,2), torch.pow(x,3), torch.pow(x,4)], dim=-1) # [B, 3D]
        x_raw     = self.preprocess(x)            # [B,H]
        x_fourier = self.hybrid(x)                # [B,2M]
        x_combined = torch.cat([x, x_poly,x_raw, x_fourier], dim=-1) # [B, 4D+H+2M]
        x = F.elu(self.linear2(x_combined))
        x = self.linear3(x)
        return x.unsqueeze(1)                     # [B,1,target_dims]

class ConditionalMarkovianTSPostMeanScoreMatchingNew(nn.Module):
    def __init__(
        self,
        max_diff_steps: int,
        diff_embed_size: int,
        diff_hidden_size: int,
        ts_dims: int,
        mlp_hidden_dims: int,
        condupsampler_length: int = 20,
        residual_layers: int = 10,
        residual_channels: int = 8,
        dilation_cycle_length: int = 10
    ):
        super().__init__()
        self.ts_dims = ts_dims
        C = residual_channels

        self.ln_in   = nn.LayerNorm(ts_dims, eps=1e-6)
        self.ln_cond = nn.LayerNorm(ts_dims, eps=1e-6)

        self.calib_scale = nn.Parameter(torch.ones(ts_dims))
        self.calib_bias  = nn.Parameter(torch.zeros(ts_dims))

        self.input_projection  = nn.Conv1d(1, C, kernel_size=1)
        self.diffusion_embedding = DiffusionEmbedding(
            diff_embed_size=diff_embed_size,
            diff_hidden_size=diff_hidden_size,
            max_steps=max_diff_steps
        )
        self.time_to_channels = nn.Linear(diff_hidden_size, C)

        self.mlp_state_mapper = MLPStateMapper(
            ts_input_dim=ts_dims, hidden_dim=mlp_hidden_dims, target_dims=condupsampler_length
        )
        self.cond_upsampler = CondUpsampler(target_dim=ts_dims, cond_length=condupsampler_length)

        # co-prime dilations (adaptive to ts_dims)
        def coprime_dilations(L: int, depth: int):
            cands = [d for d in range(1, L) if math.gcd(d, L) == 1]
            if not cands: cands = [1]
            return [cands[i % len(cands)] for i in range(depth)]

        dils = coprime_dilations(ts_dims, residual_layers)
        self.residual_layers = nn.ModuleList([ResidualBlock(C, d) for d in dils])
        C = residual_channels
        self.token_mixers = nn.ModuleList([TokenISAB(C=C, m=64, heads=4, p=0.0) for _ in self.residual_layers])

        self.skip_projection   = nn.Conv1d(C, C, kernel_size=1)
        self.output_projection = nn.Conv1d(C, 1, kernel_size=1)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, inputs, times, conditioner, eff_times):
        # x: [B,1,D] -> [B,C,D]
        x = self.input_projection(inputs)
        x = self.ln_in(x)
        x = F.silu(x)

        # build per-sample time bias [B,C,1]
        B = inputs.size(0)
        tvec = times.reshape(B, -1)[:, 0]                 # robust to [B,T,1] etc.
        temb = self.diffusion_embedding(tvec)             # [B, diff_hidden_size]
        time_bias = self.time_to_channels(temb).unsqueeze(-1)  # [B,C,1]

        # conditioner: [B,1,D] -> upsampled [B,1,D]
        cond = self.mlp_state_mapper(conditioner)
        cond_up = self.cond_upsampler(cond)
        cond_up = self.ln_cond(cond_up)

        # sanity checks
        C = self.input_projection.out_channels
        assert x.size(1) == C and time_bias.size(1) == C and cond_up.size(1) == 1
        for i, blk in enumerate(self.residual_layers):
            assert blk.dilated_conv.in_channels == C, f"block {i} has {blk.dilated_conv.in_channels}!={C}"

        # residual stack
        skip = []
        for i in range(len(self.residual_layers)):
            x, s = self.residual_layers[i](x, conditioner=cond_up, time_bias=time_bias)  # always runs
            x = self.token_mixers[i](x)  # always runs
            x = F.silu(x)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x); x = F.silu(x)
        x = self.output_projection(x)

        # per-dim calibration
        x = x * self.calib_scale.view(1, 1, -1) + self.calib_bias.view(1, 1, -1)

        # VPSDE mapping to score
        beta_tau   = torch.exp(-0.5 * eff_times)
        sigma2_tau = (1. - torch.exp(-eff_times))
        return -inputs / sigma2_tau + (beta_tau / sigma2_tau) * x
