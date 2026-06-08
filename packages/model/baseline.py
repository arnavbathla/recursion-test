"""Non-recursive baseline.

Same input packing and output heads as ``TRMARCModel``, but with a fixed stack of
*distinct* layers applied in a single forward pass -- no weight reuse, no latent
state carried across recursive steps. This isolates the contribution of recursive
refinement. The forward signature is kept drop-in compatible with ``TRMARCModel``
(it accepts and ignores ``recursion_steps``/carried states) so the same training,
eval, decode, and serving code paths work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from packages.model.trm_arc import (
    MAX_GRID,
    VOCAB_SIZE,
    RecursiveCell,
    TRMARCOutput,
)


class BaselineARCModel(nn.Module):
    """Single-pass, non-recursive ARC model with distinct per-layer weights."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        max_seq_len: int = 4096,
        num_segments: int = 8,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.recursive = False
        self.config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "mlp_dim": mlp_dim,
            "max_seq_len": max_seq_len,
            "num_segments": num_segments,
            "num_layers": num_layers,
            "dropout": dropout,
        }

        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.row_emb = nn.Embedding(MAX_GRID + 2, d_model)
        self.col_emb = nn.Embedding(MAX_GRID + 2, d_model)
        self.segment_emb = nn.Embedding(num_segments, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Distinct (non-shared) layers -> no recursive weight reuse.
        self.layers = nn.ModuleList(
            [
                RecursiveCell(d_model=d_model, n_heads=n_heads, mlp_dim=mlp_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.answer_layers = nn.ModuleList(
            [
                RecursiveCell(d_model=d_model, n_heads=n_heads, mlp_dim=mlp_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.height_head = nn.Linear(d_model, MAX_GRID)
        self.width_head = nn.Linear(d_model, MAX_GRID)
        self.cell_head = nn.Linear(d_model, 10)

    def encode(self, tokens, rows, cols, segments) -> torch.Tensor:
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, seq_len)
        return (
            self.token_emb(tokens)
            + self.row_emb(rows.clamp(0, MAX_GRID + 1))
            + self.col_emb(cols.clamp(0, MAX_GRID + 1))
            + self.segment_emb(segments)
            + self.pos_emb(pos)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        segments: torch.Tensor,
        answer_positions: torch.Tensor,
        recursion_steps: int = 1,  # ignored; baseline is single-pass
        latent_state: torch.Tensor | None = None,  # ignored
        answer_state: torch.Tensor | None = None,  # ignored
        attention_mask: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> TRMARCOutput | tuple[TRMARCOutput, list[TRMARCOutput]]:
        x = self.encode(tokens, rows, cols, segments)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        bsz, num_slots = answer_positions.shape
        gather_idx = answer_positions.unsqueeze(-1).expand(bsz, num_slots, self.d_model)
        ans = torch.gather(x, dim=1, index=gather_idx)
        for layer in self.answer_layers:
            ans = layer(ans, attention_mask=None)

        pooled = ans.mean(dim=1)
        out = TRMARCOutput(
            height_logits=self.height_head(pooled),
            width_logits=self.width_head(pooled),
            cell_logits=self.cell_head(ans),
            latent_state=x,
            answer_state=ans,
        )
        if return_trace:
            return out, [out]
        return out
