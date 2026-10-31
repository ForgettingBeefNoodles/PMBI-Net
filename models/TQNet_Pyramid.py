"""Bidirectional temporal pyramid around the UNCHANGED local TQNet backbone.

New stochastic blocks preserve the backbone RNG stream. Thus adding zero-output
branches preserves both shared initialization and training/dropout equivalence.
Only four patches of ONE variable attend to one another in the local blocks.
"""
from contextlib import contextmanager
import torch
from torch import nn
from models.TQNet import Model as TQNet

VARIANTS = ('baseline', 'patch_only', 'bottom_only', 'top_only', 'full',
            'no_phase', 'no_patch_attention', 'parallel')


@contextmanager
def preserve_rng(x=None):
    devices = [x.device.index] if x is not None and x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        yield


def mlp(n_in, n_out, r, expansion, dropout):
    return nn.Sequential(nn.Linear(n_in, expansion * r), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(expansion * r, n_out))


class Model(TQNet):
    def __init__(self, configs):
        super().__init__(configs)
        self.variant = getattr(configs, 'pyramid_variant', 'full')
        if self.variant not in VARIANTS:
            raise ValueError(f'Unknown pyramid variant: {self.variant}')
        if self.variant == 'baseline':
            return
        self.patch_len = getattr(configs, 'pyramid_patch_len', 24)
        self.r = r = getattr(configs, 'pyramid_dim', 128)
        heads = getattr(configs, 'pyramid_heads', 4)
        expansion = getattr(configs, 'pyramid_expansion', 2)
        dropout = getattr(configs, 'pyramid_dropout', .1)
        if self.seq_len % self.patch_len or r % heads:
            raise ValueError('seq_len must divide into patches; pyramid_dim must divide by heads')
        self.npatch = self.seq_len // self.patch_len
        self.bottom = self.variant in ('bottom_only', 'full', 'no_phase', 'no_patch_attention')
        self.top = self.variant in ('top_only', 'full', 'no_phase', 'no_patch_attention')
        self.phase = self.top and self.variant != 'no_phase'
        self.local_attention = self.variant != 'no_patch_attention' and bool(
            getattr(configs, 'pyramid_patch_attention', 1))
        # super() has already initialized every original parameter. Additional
        # random draws must not move the caller's original RNG stream.
        with preserve_rng():
            self.patch_encoder = nn.Sequential(nn.Linear(self.patch_len, r), nn.LayerNorm(r), nn.GELU())
            self.patch_position = nn.Parameter(torch.empty(1, 1, self.npatch, r))
            nn.init.normal_(self.patch_position, std=.02)
            if self.local_attention:
                self.patch_attention = nn.MultiheadAttention(r, heads, dropout=dropout, batch_first=True)
                self.patch_norm = nn.LayerNorm(r)
                self.patch_dropout = nn.Dropout(dropout)
            if self.variant != 'patch_only':
                self.window_encoder = nn.Sequential(nn.Linear(self.seq_len, r), nn.LayerNorm(r), nn.GELU())
            if self.bottom:
                self.bottom_attention = nn.MultiheadAttention(r, heads, dropout=dropout, batch_first=True)
                self.bottom_fusion = mlp(2*r, r, r, expansion, dropout)
                self.bottom_norm = nn.LayerNorm(r)
                self.query_delta = nn.Linear(r, self.seq_len)
                nn.init.zeros_(self.query_delta.weight)
                nn.init.zeros_(self.query_delta.bias)
            if self.phase:
                self.phase_encoder = nn.Linear(self.patch_len, r)
            if self.top:
                self.global_encoder = nn.Linear(self.seq_len, r)
                self.top_window = mlp(2*r, r, r, expansion, dropout)
                self.top_norm = nn.LayerNorm(r)
                self.top_patch = mlp((3 if self.phase else 2)*r, r, r, expansion, dropout)
            if self.variant == 'parallel':
                self.parallel_period = nn.Sequential(nn.Linear(self.seq_len, r), nn.LayerNorm(r), nn.GELU())
                self.parallel_fusion = mlp((self.npatch+2)*r, self.npatch*r, r, expansion, dropout)
            if self.variant != 'bottom_only':
                self.patch_readout = nn.Linear(self.npatch*r, self.seq_len)
                nn.init.zeros_(self.patch_readout.weight)
                nn.init.zeros_(self.patch_readout.bias)

    def phase_indices(self, cycle_index):
        if cycle_index.ndim != 1:
            raise ValueError('cycle_index must be [B], the FIRST forecast position')
        offsets = torch.arange(self.seq_len, device=cycle_index.device)
        future = (cycle_index.long()[:, None] + offsets) % self.cycle_len
        history = (cycle_index.long()[:, None] - self.seq_len + offsets) % self.cycle_len
        return future, history

    def forward(self, x, cycle_index, return_details=False):
        if self.variant == 'baseline':
            return super().forward(x, cycle_index)
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f'Expected [B,{self.seq_len},{self.enc_in}], got {tuple(x.shape)}')
        if self.use_revin:
            seq_mean = x.mean(dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x-seq_mean) / torch.sqrt(seq_var)
        xc = x.permute(0, 2, 1)
        b, c, _ = xc.shape
        future, history = self.phase_indices(cycle_index)
        q0 = self.temporalQuery[future].permute(0, 2, 1)
        details = {}
        with preserve_rng(x):
            h24 = self.patch_encoder(xc.reshape(b, c, self.npatch, self.patch_len)) + self.patch_position
            if self.local_attention:
                p = h24.reshape(b*c, self.npatch, self.r)
                a = self.patch_attention(p, p, p, need_weights=False)[0]
                h24 = self.patch_norm(p+self.patch_dropout(a)).reshape(b, c, self.npatch, self.r)
            h96 = z96 = self.window_encoder(xc) if self.variant != 'patch_only' else None
            q = q0
            if self.bottom:
                a = self.bottom_attention(z96.reshape(b*c, 1, self.r),
                    h24.reshape(b*c, self.npatch, self.r),
                    h24.reshape(b*c, self.npatch, self.r), need_weights=False)[0].reshape(b, c, self.r)
                h96 = self.bottom_norm(z96+self.bottom_fusion(torch.cat((z96, a), dim=-1)))
                q = q0+self.query_delta(h96)
        # Keep original need_weights=True, heads=4, dropout=.5 and all Q/K/V
        # projections. No other cross-variable attention is executed.
        global_ = self.channelAggregator(query=q, key=xc, value=xc)[0]
        with preserve_rng(x):
            h24_top = h24
            if self.top:
                g96 = self.global_encoder(global_)
                h96_top = self.top_norm(h96+self.top_window(torch.cat((h96, g96), dim=-1)))
                conditions = [h24, h96_top.unsqueeze(2).expand(-1, -1, self.npatch, -1)]
                if self.phase:
                    p_hist = self.temporalQuery[history].permute(0, 2, 1)
                    phase24 = self.phase_encoder(p_hist.reshape(b, c, self.npatch, self.patch_len))
                    conditions.append(phase24)
                    details['Phase24'] = phase24
                h24_top = h24+self.top_patch(torch.cat(conditions, dim=-1))
                details['H96_top'] = h96_top
            if self.variant == 'parallel':
                # Independent three-scale encoding; no sequential bottom/top
                # interaction, and the original periodic query stays unchanged.
                period = self.parallel_period(q0)
                h24_top = self.parallel_fusion(torch.cat((h24.flatten(2), z96, period), dim=-1))
            residual = self.patch_readout(h24_top.flatten(2)) if self.variant != 'bottom_only' else 0
        inp = self.input_proj(xc+global_+residual)
        output = self.output_proj(self.model(inp)+inp).permute(0, 2, 1)
        if self.use_revin:
            output = output*torch.sqrt(seq_var)+seq_mean
        if return_details:
            details.update(H24=h24, Z96=z96, H96=h96, Q0=q0, Q_modified=q,
                           Global=global_, H24_top=h24_top, R_pyramid=residual,
                           future_phase=future, history_phase=history)
            return output, details
        return output

    def parameter_breakdown(self):
        groups = {}
        for name, p in self.named_parameters():
            key = name.split('.')[0]
            groups[key] = groups.get(key, 0)+p.numel()
        return groups
