"""Unmodified helper implementations extracted from the local TQNet Pyramid code."""
from contextlib import contextmanager
import torch
from torch import nn

@contextmanager
def preserve_rng(x=None):
    devices = [x.device.index] if x is not None and x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        yield

def mlp(n_in, n_out, r, expansion, dropout):
    return nn.Sequential(nn.Linear(n_in, expansion * r), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(expansion * r, n_out))

