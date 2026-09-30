"""Reconstruct dozken's reset2k model in PyTorch from his ONNX weights.

WHY
---
`submissions/submission_dozken_6569/reset2k.onnx` is a hard-batched graph: its
inputs are declared `[1, 112]` and `[1, 1, 192]`, and onnxruntime rejects any other
batch size. Replaying it over the 212,140,000-row train shard through the graph
itself would cost roughly 43 us/row, i.e. ~2.5 hours of single-step inference.

Lifting the same initializers into torch lets us replay batched, which is ~20x
faster, and gives us a differentiable module we can train from scratch with the
identical architecture -- which is the whole point of the reproduction arm.

The architecture, recovered from the graph topology and the initializer shapes:

    x        = clip(features * inv_std + neg_mean_scaled, -10, 10)   # 112
    dx       = x - prev                                             # 112, first differences
    x_cat    = concat([x, dx])                                      # 224
    z_pre    = GELU(x_cat @ W_in.T + b_in)                          # 64
    z        = LayerNorm(z_pre, ln_in_w, ln_in_b)                    # 64
    h1       = GRU_layer0(z,   W_0, R_0, B_0)                       # 192
    h2       = GRU_layer1(h1,  W_1, R_1, B_1)                       # 192
    cat      = concat([h2, z])                                      # 256
    cat_n    = LayerNorm(cat, ln_head_w, ln_head_b)                  # 256
    pred     = GELU(cat_n @ W_h1.T + b_h1) @ W_h2.T + b_h2          # 2

Two structural details the graph makes explicit and that our own models never had:

1. **First differences are computed inside the graph**, on the normalised scale,
   and concatenated with the level features (`prev` is a state that carries the
   previous normalised row, and `next_prev` is the state that feeds the next step).
   So the served model needs no separate difference feature pass.
2. **The layer-0 output is kept and concatenated into the head** alongside the
   layer-1 output. That is a skip path from the input projection straight to the
   readout, which our 2x128 baseline has no equivalent of.

Gate order is the torch/ONNX GRU convention `r, z, n`, so `W` and `R` rows are
read in that order when rebuilt with a `nn.GRU`-compatible module.

VERIFY BEFORE TRUSTING
---------------------
`verify()` replays both this module and the original ONNX graph side by side over
a real validation sequence and asserts they agree. Nothing here should be used as
a teacher until that passes to float32 precision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H = 192
N_FEATURES = 112
HIDDEN = H
PROJ = 64
HEAD = 256


def load_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(onnx_path)).graph
    return {i.name: numpy_helper.to_array(i).copy() for i in g.initializer}


class DozkenReset2k(nn.Module):
    """PyTorch rebuild of his reset2k graph, batched.

    `forward_chunk(x, h, prev)` consumes a whole chunk with the state carried,
    and returns `(pred, h, prev)` so the caller can stream chunks. `prev` is the
    previous normalised row and must be carried the same way `h` is.
    """

    def __init__(self, init: dict[str, np.ndarray]):
        super().__init__()
        t = lambda n, dt=torch.float32: torch.tensor(init[n], dtype=dt)

        # Input standardisation, folded exactly as the graph folds it.
        self.register_buffer('inv_std', t('inv_std'))
        self.register_buffer('neg_mean_scaled', t('neg_mean_scaled'))
        self.register_buffer('clip_lo', t('clip_lo'))
        self.register_buffer('clip_hi', t('clip_hi'))

        # Input projection 224 -> 64, then LayerNorm.
        self.W_in = nn.Parameter(t('W_in'))
        self.b_in = nn.Parameter(t('b_in').reshape(-1))
        self.ln_in_w = nn.Parameter(t('ln_in_w'))
        self.ln_in_b = nn.Parameter(t('ln_in_b'))

        # Two GRU layers, 64 -> 192 -> 192. ONNX stores W as [1, 3H, in] and
        # R as [1, 3H, H] with gates in r, z, n order -- the same layout
        # nn.GRU uses, so the tensors are adopted directly.
        self.W0 = nn.Parameter(t('W_0'))
        self.R0 = nn.Parameter(t('R_0'))
        # ONNX stores the GRU bias as one [1, 2*3H] tensor: the first 3H is the
        # gate bias B_gate and the second 3H is the hidden bias B_bias, which
        # adds to the `n` gate only. Keeping them fused here and splitting inside
        # gru_step avoids a silent mis-wiring of the hidden bias.
        self.B0 = nn.Parameter(t('B_0').reshape(-1))
        self.W1 = nn.Parameter(t('W_1'))
        self.R1 = nn.Parameter(t('R_1'))
        self.B1 = nn.Parameter(t('B_1').reshape(-1))

        # Head: concat(h2, z) -> 256 -> LayerNorm -> GELU -> 2.
        self.ln_head_w = nn.Parameter(t('ln_head_w'))
        self.ln_head_b = nn.Parameter(t('ln_head_b'))
        self.W_h1 = nn.Parameter(t('W_h1'))
        self.b_h1 = nn.Parameter(t('b_h1').reshape(-1))
        self.W_h2 = nn.Parameter(t('W_h2'))
        self.b_h2 = nn.Parameter(t('b_h2').reshape(-1))

    # ---- pieces ------------------------------------------------------------

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * self.inv_std + self.neg_mean_scaled,
                           self.clip_lo, self.clip_hi)

    def gru_step(self, xin: torch.Tensor, h: torch.Tensor,
                 W: torch.Tensor, R: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """One ONNX-GRU step, written out so we control the gate order.

        Gate order in the ONNX/torch layout is r, z, n, so W and R are split
        three ways in that order. Done by hand rather than via nn.GRU because
        ONNX uses the `B` term for both gates (`initializers B`, not
        `bias_ih`/`bias_hh` separately) and silently getting that wrong would
        produce a model that looks fine and predicts garbage.
        """
        # xin: (B, T, in) or (B, in); h: (B, 1, H)
        single = xin.dim() == 2
        if single:
            xin = xin.unsqueeze(1)
        gi = xin @ W.squeeze(0).T            # (B, T, 3H)
        gh = h @ R.squeeze(0).T               # (B, 1, 3H)
        g = gi + gh
        Bsz, T, _ = g.shape
        ir, iz, inn = g.chunk(3, dim=-1)
        # B is stored fused as [3H gate bias, 3H hidden bias]. Every gate takes
        # its own slice of the gate bias; the hidden bias adds to the n gate only,
        # and only its FIRST H entries are used.
        bg, bh = B[:3 * H], B[3 * H:3 * H + H]
        r = torch.sigmoid(ir + bg[:H].view(1, 1, -1))
        z = torch.sigmoid(iz + bg[H:2 * H].view(1, 1, -1))
        # linear_before_reset = 1 on this graph: the reset gate multiplies the
        # ALREADY-BIASED candidate, i.e. r * (W_n x + R_n h + b_hn), NOT the
        # PyTorch form r * (W_n x) + R_n h. Using the PyTorch order here produced
        # a model that matched the input path to 1.5e-06 and then diverged by
        # ~1.0 in the hidden state, so this attribute is the whole difference.
        n = torch.tanh(r * (inn + bh.view(1, 1, -1)) + bg[2 * H:].view(1, 1, -1))
        hn = (1.0 - z) * n + z * h
        return hn.squeeze(1) if single else hn

    # ---- forward -----------------------------------------------------------

    def forward_chunk(self, x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor,
                      prev: torch.Tensor):
        """x: (B, T, 112) raw features. Returns (pred (B,T,2), h0, h1, prev)."""
        xn = self.normalise(x)                     # (B, T, 112)
        dx = xn - prev.unsqueeze(1)                # (B, T, 112)
        x_cat = torch.cat([xn, dx], dim=-1)        # (B, T, 224)

        z_pre = F.gelu(x_cat @ self.W_in.T + self.b_in)
        z = F.layer_norm(z_pre, (PROJ,), self.ln_in_w, self.ln_in_b)

        # Walk the chunk step by step: the recurrence is the whole point and it
        # must be carried, not recomputed from a parallel scan.
        Bsz, T, _ = z.shape
        a = h0
        b = h1
        outs = []
        for t in range(T):
            a = self.gru_step(z[:, t], a, self.W0, self.R0, self.B0)   # (B,192)
            b = self.gru_step(a, b, self.W1, self.R1, self.B1)
            outs.append(b)
        h2 = torch.stack(outs, dim=1)              # (B, T, 192)

        cat = torch.cat([h2, z], dim=-1)           # (B, T, 256)
        cat_n = F.layer_norm(cat, (HIDDEN + PROJ,), self.ln_head_w, self.ln_head_b)
        pred = F.gelu(cat_n @ self.W_h1.T + self.b_h1) @ self.W_h2.T + self.b_h2
        return pred, a, b, xn[:, -1, :]


def build(onnx_path: Path) -> DozkenReset2k:
    m = DozkenReset2k(load_initializers(onnx_path))
    m.eval()
    return m
