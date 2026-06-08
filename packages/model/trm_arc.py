"""TRM/HRM-inspired latent-recursive ARC model.

This is the center of the system. Recursive reasoning here means repeatedly
applying the *same* weights while carrying a persistent latent reasoning state and
a persistent answer state forward across steps -- not chain-of-thought in token
space. There is no LLM dependency and no hardcoded ARC rule; the model only sees
grid tokens + positional/segment signals, never a task id.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_GRID = 30
MAX_CELLS = MAX_GRID * MAX_GRID
VOCAB_SIZE = 19
PAD = 10


@dataclass
class TRMARCOutput:
    height_logits: torch.Tensor       # [B, 30]
    width_logits: torch.Tensor        # [B, 30]
    cell_logits: torch.Tensor         # [B, 900, 10]
    latent_state: torch.Tensor        # [B, S, D]
    answer_state: torch.Tensor        # [B, 900, D]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RecursiveCell(nn.Module):
    """Shared recurrent cell.

    The same weights are applied repeatedly. This is the recursive-compute
    mechanism at the heart of the model.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_attn = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_dim)

    def forward(
        self,
        state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.norm_attn(state)

        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=attention_mask,
            need_weights=False,
        )

        state = state + attn_out
        state = state + self.mlp(self.norm_mlp(state))

        return state


class TRMARCModel(nn.Module):
    """TRM-style recursive ARC solver.

    Inputs:
      tokens:           [B, S]
      rows:             [B, S]
      cols:             [B, S]
      segments:         [B, S]
      answer_positions: [B, 900]
      attention_mask:   [B, S] bool where True means padding

    Internal states:
      latent_state = hidden reasoning scratchpad over packed task tokens
      answer_state = latent current answer proposal over the 30x30 output canvas

    Outputs:
      height_logits: [B, 30]
      width_logits:  [B, 30]
      cell_logits:   [B, 900, 10]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        max_seq_len: int = 4096,
        num_segments: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # config echo for checkpointing/registry
        self.config = {
            "d_model": d_model,
            "n_heads": n_heads,
            "mlp_dim": mlp_dim,
            "max_seq_len": max_seq_len,
            "num_segments": num_segments,
            "dropout": dropout,
        }

        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.row_emb = nn.Embedding(MAX_GRID + 2, d_model)
        self.col_emb = nn.Embedding(MAX_GRID + 2, d_model)
        self.segment_emb = nn.Embedding(num_segments, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.net = RecursiveCell(
            d_model=d_model,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.answer_net = RecursiveCell(
            d_model=d_model,
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.answer_to_latent = nn.Linear(d_model, d_model)

        self.height_head = nn.Linear(d_model, MAX_GRID)
        self.width_head = nn.Linear(d_model, MAX_GRID)
        self.cell_head = nn.Linear(d_model, 10)

    def encode(
        self,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        segments: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = tokens.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        pos = torch.arange(seq_len, device=tokens.device)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)

        return (
            self.token_emb(tokens)
            + self.row_emb(rows.clamp(0, MAX_GRID + 1))
            + self.col_emb(cols.clamp(0, MAX_GRID + 1))
            + self.segment_emb(segments)
            + self.pos_emb(pos)
        )

    def gather_answer_context(
        self,
        latent_state: torch.Tensor,
        answer_positions: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_slots = answer_positions.shape
        dim = latent_state.size(-1)
        gather_idx = answer_positions.unsqueeze(-1).expand(bsz, num_slots, dim)
        return torch.gather(latent_state, dim=1, index=gather_idx)

    def initialize_states(
        self,
        encoded_task: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = encoded_task.shape

        latent_state = torch.zeros_like(encoded_task)
        answer_state = torch.zeros(
            bsz,
            MAX_CELLS,
            dim,
            device=encoded_task.device,
            dtype=encoded_task.dtype,
        )

        return latent_state, answer_state

    def recursive_step(
        self,
        encoded_task: torch.Tensor,
        latent_state: torch.Tensor,
        answer_state: torch.Tensor,
        answer_positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        answer_summary = answer_state.mean(dim=1, keepdim=True)
        answer_feedback = self.answer_to_latent(answer_summary)
        answer_feedback = answer_feedback.expand(-1, encoded_task.size(1), -1)

        latent_input = encoded_task + latent_state + answer_feedback
        latent_state = self.net(latent_input, attention_mask=attention_mask)

        answer_context = self.gather_answer_context(latent_state, answer_positions)
        answer_state = self.answer_net(answer_state + answer_context)

        return latent_state, answer_state

    def readout(
        self,
        answer_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = answer_state.mean(dim=1)

        height_logits = self.height_head(pooled)
        width_logits = self.width_head(pooled)
        cell_logits = self.cell_head(answer_state)

        return height_logits, width_logits, cell_logits

    def forward(
        self,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        segments: torch.Tensor,
        answer_positions: torch.Tensor,
        recursion_steps: int = 16,
        latent_state: torch.Tensor | None = None,
        answer_state: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> TRMARCOutput | tuple[TRMARCOutput, list[TRMARCOutput]]:
        if recursion_steps < 1:
            raise ValueError(f"recursion_steps must be >= 1, got {recursion_steps}")

        encoded_task = self.encode(tokens, rows, cols, segments)

        if latent_state is None or answer_state is None:
            latent_state, answer_state = self.initialize_states(encoded_task)

        trace: list[TRMARCOutput] = []
        trace_steps = {1, 2, 4, 8, 16, 32, 64}

        for step in range(1, recursion_steps + 1):
            latent_state, answer_state = self.recursive_step(
                encoded_task=encoded_task,
                latent_state=latent_state,
                answer_state=answer_state,
                answer_positions=answer_positions,
                attention_mask=attention_mask,
            )

            if return_trace and step in trace_steps:
                h_logits, w_logits, c_logits = self.readout(answer_state)
                trace.append(
                    TRMARCOutput(
                        height_logits=h_logits,
                        width_logits=w_logits,
                        cell_logits=c_logits,
                        latent_state=latent_state,
                        answer_state=answer_state,
                    )
                )

        height_logits, width_logits, cell_logits = self.readout(answer_state)

        output = TRMARCOutput(
            height_logits=height_logits,
            width_logits=width_logits,
            cell_logits=cell_logits,
            latent_state=latent_state,
            answer_state=answer_state,
        )

        if return_trace:
            return output, trace

        return output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
