"""PMBI-Net: Periodic Prior-Guided Multi-Granularity Bidirectional Interaction Network.

The formal v2_full implementation is defined here (not a wrapper around V2).
MTFE: patch_encoder, window_encoder, period_encoder.
HBI: bottom_attention / fusion_24_96 / fusion_96_p and fusion_p_96 /
feedback_96_24, with historical phase_encoder feedback.
Joint Readout: [UP, D96, flatten(D24)] -> original temporal width.

TQNet inheritance and every model operation/initialization are retained verbatim.
Legacy variants remain in the reference-compatible class; run.py exposes Full only.
"""
import torch
from torch import nn
from models.TQNet import Model as OriginalTQNet
from utils.pmbi_helpers import preserve_rng, mlp

VARIANTS = ('baseline', 'v2_full', 'v2_parallel', 'v2_bottom', 'v2_top',
            'v2_no_phase', 'v2_no_patch_attention')


class Model(OriginalTQNet):
    def __init__(self, configs):
        super().__init__(configs)
        self.variant = getattr(configs, 'pyramid_v2_variant', 'v2_full')
        if self.variant == 'baseline':
            raise ValueError('V2 baseline must use model=TQNet directly')
        if self.variant not in VARIANTS:
            raise ValueError('Invalid V2 variant')
        self.r = r = configs.pyramid_dim
        self.patch_len = configs.pyramid_patch_len
        if self.seq_len != 96 or self.patch_len != 24 or r % configs.pyramid_heads:
            raise ValueError('V2 uses L96 / four24-step patches; r must divide by local heads')
        self.npatch = 4
        self.bottom = self.variant not in ('v2_parallel', 'v2_top')
        self.top = self.variant not in ('v2_parallel', 'v2_bottom')
        self.phase = self.top and self.variant != 'v2_no_phase'
        self.local_attention = self.variant != 'v2_no_patch_attention' and bool(configs.pyramid_patch_attention)
        heads, expansion, dropout = configs.pyramid_heads, configs.pyramid_expansion, configs.pyramid_dropout
        with preserve_rng():
            self.patch_encoder = nn.Sequential(nn.Linear(24,r),nn.LayerNorm(r),nn.GELU())
            self.patch_position = nn.Parameter(torch.empty(1,1,4,r))
            nn.init.normal_(self.patch_position,std=.02)
            if self.local_attention:
                self.patch_attention = nn.MultiheadAttention(r,heads,dropout=dropout,batch_first=True)
                self.patch_norm = nn.LayerNorm(r)
                self.patch_dropout = nn.Dropout(dropout)
            self.window_encoder = nn.Sequential(nn.Linear(96,r),nn.LayerNorm(r),nn.GELU())
            self.period_encoder = nn.Linear(96,r)
            if self.bottom:
                self.bottom_attention = nn.MultiheadAttention(r,heads,dropout=dropout,batch_first=True)
                self.fusion_24_96 = mlp(2*r,r,r,expansion,dropout)
                self.bottom_window_norm = nn.LayerNorm(r)
                self.fusion_96_p = mlp(2*r,r,r,expansion,dropout)
                self.bottom_period_norm = nn.LayerNorm(r)
            if self.top:
                self.fusion_p_96 = mlp(2*r,r,r,expansion,dropout)
                self.top_window_norm = nn.LayerNorm(r)
                if self.phase:
                    self.phase_encoder = nn.Linear(24,r)
                self.feedback_96_24 = mlp((3 if self.phase else 2)*r,r,r,expansion,dropout)
            # All three final scales feed one nonlinear readout. Only the LAST
            # output projection is zero; there is no second zero multiplier.
            self.multiscale_readout = mlp((2+self.npatch)*r,self.seq_len,r,expansion,dropout)
            nn.init.zeros_(self.multiscale_readout[-1].weight)
            nn.init.zeros_(self.multiscale_readout[-1].bias)

    def forward(self,x,cycle_index,return_details=False):
        if x.shape[1:] != (self.seq_len,self.enc_in) or cycle_index.shape != (x.shape[0],):
            raise ValueError('Expected x[B,96,C] and forecast-start cycle_index[B]')
        if self.use_revin:
            mean=x.mean(1,keepdim=True)
            var=torch.var(x,dim=1,keepdim=True)+1e-5
            x=(x-mean)/torch.sqrt(var)
        xc=x.permute(0,2,1)
        b,c,_=xc.shape
        offsets=torch.arange(self.seq_len,device=cycle_index.device)
        future=(cycle_index.long()[:,None]+offsets)%self.cycle_len
        history=(cycle_index.long()[:,None]-self.seq_len+offsets)%self.cycle_len
        q0=self.temporalQuery[future].permute(0,2,1)
        # Independent initial patch and full-window representations.
        with preserve_rng(x):
            z24=self.patch_encoder(xc.reshape(b,c,4,24))+self.patch_position
            if self.local_attention:
                patch=z24.reshape(b*c,4,self.r)
                attended=self.patch_attention(patch,patch,patch,need_weights=False)[0]
                z24=self.patch_norm(patch+self.patch_dropout(attended)).reshape(b,c,4,self.r)
            z96=self.window_encoder(xc)
        # EXACT original Q/K/V, heads4, dropout.5, default attention call flags.
        # No bottom-up operation has executed at this point.
        g=self.channelAggregator(query=q0,key=xc,value=xc)[0]
        with preserve_rng(x):
            zp=self.period_encoder(g)
            u96,up=z96,zp
            if self.bottom:
                a=self.bottom_attention(z96.reshape(b*c,1,self.r),
                    z24.reshape(b*c,4,self.r),z24.reshape(b*c,4,self.r),need_weights=False)[0].reshape(b,c,self.r)
                u96=self.bottom_window_norm(z96+self.fusion_24_96(torch.cat((z96,a),-1)))
                up=self.bottom_period_norm(zp+self.fusion_96_p(torch.cat((zp,u96),-1)))
            d96,d24=u96,z24
            phase24=None
            if self.top:
                d96=self.top_window_norm(u96+self.fusion_p_96(torch.cat((u96,up),-1)))
                conditions=[z24,d96.unsqueeze(2).expand(-1,-1,4,-1)]
                if self.phase:
                    phase24=self.phase_encoder(self.temporalQuery[history].permute(0,2,1).reshape(b,c,4,24))
                    conditions.append(phase24)
                d24=z24+self.feedback_96_24(torch.cat(conditions,-1))
            residual=self.multiscale_readout(torch.cat((up,d96,d24.flatten(2)),-1))
        inp=self.input_proj(xc+g+residual)
        out=self.output_proj(self.model(inp)+inp).permute(0,2,1)
        if self.use_revin:
            out=out*torch.sqrt(var)+mean
        if return_details:
            return out,dict(Xc=xc,Q0=q0,G=g,Z24=z24,Z96=z96,ZP=zp,U96=u96,UP=up,
                D96=d96,D24=d24,R=residual,Phase24=phase24,future_index=future,history_index=history)
        return out

    def parameter_breakdown(self):
        result={}
        for name,p in self.named_parameters():
            key=name.split('.')[0]
            result[key]=result.get(key,0)+p.numel()
        return result
