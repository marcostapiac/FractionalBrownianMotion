import math
import torch
import torch.nn.functional as F
from torch import nn

# ---------- utils ----------
@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)

def _causal_left_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    # x: [B, C, T] -> left-pad by 'pad' with zeros (causal conv)
    if pad <= 0:
        return x
    return F.pad(x, (pad, 0))

# ---------- embeddings ----------
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
        if diffusion_step.dtype in (torch.int32, torch.int64):
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = silu(self.projection1(x))
        x = silu(self.projection2(x))
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (t - low_idx).unsqueeze(-1) * (high - low)

    @staticmethod
    def _build_embedding(diff_embed_size: int, max_steps: int):
        steps = torch.arange(max_steps).unsqueeze(1)            # [S,1]
        dims = torch.arange(diff_embed_size).unsqueeze(0)       # [1,E]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)             # [S,E]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [S,2E]
        return table

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int):
    # (kept for parity)
    timesteps = timesteps.view(timesteps.shape[0], )
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor([10000.0])) / (half_dim - 1)
    emb = torch.exp(torch.arange(start=0, end=half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.to(torch.float32)[:, None] * emb[None, :]
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    assert emb.shape[0] == timesteps.shape[0] and emb.shape[1] == embedding_dim
    return emb

# ---------- residual TCN ----------
class ResidualBlock(nn.Module):
    def __init__(self, diffusion_hidden_size, residual_channels, dilation, kernel_size=3):
        super().__init__()
        self.dilation = dilation
        self.k = kernel_size
        self.left_pad = (self.k - 1) * dilation

        # Causal conv (manual left pad, so padding=0)
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size=self.k,
            padding=0,
            dilation=dilation
        )
        # Conditioner expects [B,1,T]
        self.conditioner_projection = nn.Conv1d(1, 2 * residual_channels, 1)
        self.diffusion_projection = nn.Linear(diffusion_hidden_size, residual_channels)

        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)
        nn.init.kaiming_normal_(self.output_projection.weight)

        # Mild channel-wise normalization to reduce gain spikes
        self.gn_in = nn.GroupNorm(1, residual_channels)
        self.gn_out = nn.GroupNorm(1, 2 * residual_channels)

    def forward(self, x, conditioner, diffusion_step):
        # x: [B,C,T], conditioner: [B,1,T], diffusion_step: [B,H]
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)  # [B,C,1]
        y = x + diffusion_step

        y = self.gn_in(y)
        y = _causal_left_pad(y, self.left_pad)
        y = self.dilated_conv(y)                           # [B,2C,T]
        y = y + self.conditioner_projection(conditioner)   # [B,2C,T]

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)         # [B,C,T]

        y = self.gn_out(self.output_projection(y))         # [B,2C,T]
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip

# ---------- cond upsampler ----------
class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        mid = max(64, 2 * target_dim)
        self.linear1 = nn.Linear(cond_length, max(32, cond_length // 2), bias=False)
        self.linear2 = nn.Linear(max(32, cond_length // 2), mid, bias=False)
        self.linear3 = nn.Linear(mid, target_dim, bias=False)

    def forward(self, x):
        # x: [B,L] or [B,1,L] -> [B,T]
        if x.dim() == 3:
            assert x.size(1) == 1
            x = x.squeeze(1)
        x = F.leaky_relu(self.linear1(x), 0.2)
        x = F.leaky_relu(self.linear2(x), 0.2)
        x = F.leaky_relu(self.linear3(x), 0.2)
        return x

# ---------- hybrid state features ----------
class HybridStates(nn.Module):
    """
    Safe Fourier features (dimension-agnostic across D∈[1,20]).
    Deterministic gates by default; enable Gumbel externally if desired.
    """
    def __init__(self, D, M, init_tau=1.0, final_tau=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, D) / math.sqrt(max(D, 1)))
        self.b = nn.Parameter(2 * math.pi * torch.rand(M))
        # Start near exp(log_scale)=1.0; clamp during training
        self.log_scale = nn.Parameter(torch.randn(M) * 0.1)
        self.min_log_scale, self.max_log_scale = -2.5, 2.5  # ~ [0.082, 12.18]
        self.gate_logits = nn.Parameter(torch.ones(2 * M))   # ~0.73 after sigmoid
        self.init_tau, self.final_tau = init_tau, final_tau
        self.tau = init_tau

    def set_tau(self, tau: float):
        self.tau = tau

    def forward(self, x):
        # x: [B,D]
        with torch.no_grad():
            self.log_scale.clamp_(self.min_log_scale, self.max_log_scale)
        scales = torch.exp(self.log_scale.clamp(-2.5, 2.5)).unsqueeze(1)
        W_scaled = scales * self.W                        # [M,D]
        proj = x @ W_scaled.T + self.b                    # [B,M]
        fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B,2M]
        g = torch.sigmoid(self.gate_logits / (self.tau if self.training else self.final_tau)).unsqueeze(0)
        return g * fourier

# ---------- conditioner mapper (tail-aware, D-agnostic) ----------
class MLPStateMapper(nn.Module):
    """
    - Per-feature running standardization (robust when D=1..20)
    - Tail-aware warps: log1p|x|, sign(x), x^2_clipped, rank-gaussianized
    - Balanced branches; Fourier kept on standardized raw x
    """
    def __init__(self, ts_input_dim: int, hidden_dim: int, target_dims: int):
        super().__init__()
        D = ts_input_dim
        M = min(64, max(16, 6 * D))  # adaptive Fourier count

        self.D = D
        # Running stats buffers
        self.register_buffer("running_mean", torch.zeros(D))
        self.register_buffer("running_var", torch.ones(D))
        self.momentum = 0.01
        self.eps = 1e-5

        self.hybrid = HybridStates(D=D, M=M)

        width = max(64, hidden_dim // 2)
        in_dim = 5 * D  # x_std, log1p|x|, sign(x), x^2_clipped, rank-gauss

        self.pre_raw = nn.Sequential(nn.Linear(in_dim, width), nn.ELU())
        self.pre_mlp = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU()
        )
        self.proj_fourier = nn.Linear(2 * M, width)

        self.linear2 = nn.Linear(in_dim + 3 * width, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, target_dims)

    @torch.no_grad()
    def _update_running_stats(self, x: torch.Tensor):
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)
        self.running_mean.lerp_(mean, self.momentum)
        self.running_var.lerp_(var, self.momentum)

    def _standardize(self, x: torch.Tensor):
        if self.training:
            self._update_running_stats(x)
        return (x - self.running_mean) / (self.running_var + self.eps).sqrt()

    def set_tau(self, tau: float):
        self.hybrid.set_tau(tau)

    def forward(self, x):
        """
        x: [B,1,D] -> [B,1,target_dims]
        """
        assert (x.ndim == 3 and x.size(1) == 1 and x.size(2) == self.D)
        x = x.squeeze(1)                     # [B,D]
        x_std = self._standardize(x)         # [B,D]

        # Tail-aware per-dim warps
        log1p_abs = torch.log1p(x_std.abs())
        sign = torch.sign(x_std)
        x2 = torch.clamp(x_std * x_std, max=9.0)
        with torch.no_grad():
            # Batchwise rank -> Gaussian via erf^{-1}; clamp u to avoid infs
            ranks = torch.argsort(torch.argsort(x_std, dim=0), dim=0).float() + 1.0
            u = ranks / (x_std.size(0) + 1.0)
            u = u.clamp(1e-3, 1 - 1e-3)
            gauss = math.sqrt(2.0) * torch.erfinv(2 * u - 1)
            if torch.any(torch.isnan(gauss)) or torch.any(torch.isinf(gauss)): raise RuntimeError

        xr = torch.cat([x_std, log1p_abs, sign, x2, gauss], dim=-1)  # [B,5D]

        raw_feat = self.pre_raw(xr)                  # [B,W]
        mlp_feat = self.pre_mlp(xr)                  # [B,W]
        fourier = self.hybrid(x_std)                 # [B,2M] (on standardized x)
        fourier_feat = F.elu(self.proj_fourier(fourier))  # [B,W]

        x_combined = torch.cat([xr, raw_feat, mlp_feat, fourier_feat], dim=-1)  # [B,5D+3W]
        h = F.elu(self.linear2(x_combined))           # [B,H]
        out = self.linear3(h)                         # [B,target_dims]
        return out.unsqueeze(1)                       # [B,1,target_dims]

# ---------- full model ----------
class NewConditionalMarkovianTSPostMeanScoreMatching(nn.Module):
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
        # Avoid conditioner bottleneck at higher D (internal uplift; API unchanged)
        eff_cond_len = max(condupsampler_length, 4 * ts_dims)

        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(
            diff_embed_size=diff_embed_size,
            diff_hidden_size=diff_hidden_size,
            max_steps=max_diff_steps
        )
        eff_hidden = mlp_hidden_dims * max(4, ts_dims // 2)

        self.mlp_state_mapper = MLPStateMapper(
            ts_input_dim=ts_dims,
            hidden_dim=eff_hidden,
            target_dims=eff_cond_len
        )
        self.cond_upsampler = CondUpsampler(
            target_dim=ts_dims,
            cond_length=eff_cond_len
        )

        # Alternate k=3 and k=5 every 3rd block to broaden receptive field
        blocks = []
        for i in range(residual_layers):
            dil = 2 ** (i % dilation_cycle_length)
            k = 5 if (i % 3 == 2) else 3
            blocks.append(
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=dil,
                    diffusion_hidden_size=diff_hidden_size,
                    kernel_size=k
                )
            )
        self.residual_layers = nn.ModuleList(blocks)

        # WeightNorm to control gain and reduce wiggles
        self.skip_projection = nn.utils.weight_norm(nn.Conv1d(residual_channels, residual_channels, 1))
        self.output_projection = (nn.Conv1d(residual_channels, 1, 1))

        nn.init.kaiming_normal_(self.input_projection.weight)
        # weight_norm reparam: init weight_v
        nn.init.kaiming_normal_(self.skip_projection.weight_v)
        nn.init.zeros_(self.output_projection.weight)
        with torch.no_grad():
            if hasattr(self.skip_projection, 'weight_g'):
                self.skip_projection.weight_g.fill_(1.0)
            if hasattr(self.output_projection, 'weight_g'):
                self.output_projection.weight_g.fill_(0.5)

    def forward(self, inputs, times, conditioner, eff_times):
        """
        inputs:     [B, 1, T]
        times:      [B] or float tensor for DiffusionEmbedding
        conditioner:[B, 1, D]
        eff_times:  [B, 1, T] or [B, T] (broadcastable)
        """
        # Input stem
        x = F.leaky_relu(self.input_projection(inputs), 0.01)   # [B,C,T]

        diffusion_step = self.diffusion_embedding(times)        # [B,H]
        conditioner = self.mlp_state_mapper(conditioner)        # [B,1,L]
        if torch.any(torch.isnan(conditioner)) or torch.any(torch.isinf(conditioner)): raise RuntimeError
        cond_up = self.cond_upsampler(conditioner)              # [B,T]
        if cond_up.dim() == 2:
            cond_up = cond_up.unsqueeze(1)                      # -> [B,1,T]

        skip = []
        for layer in self.residual_layers:
            x, s = layer(x, conditioner=cond_up, diffusion_step=diffusion_step)
            x = F.leaky_relu(x, 0.01)
            skip.append(s)
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)): raise RuntimeError

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = F.leaky_relu(self.skip_projection(x), 0.01)
        x = self.output_projection(x)
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)): raise RuntimeError

        # VPSDE posterior mean target (numerically stabilized; no loss change)
        beta_tau = torch.exp(-0.5 * eff_times)
        sigma2_tau = (1.0 - torch.exp(-eff_times))
        return -inputs / sigma2_tau + (beta_tau / sigma2_tau) * x
