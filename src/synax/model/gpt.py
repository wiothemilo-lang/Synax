"""Llama-style decoder-only Transformer: RoPE, RMSNorm, SwiGLU, optional GQA."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from synax.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x).to(dtype)


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)[None, None, :, :]
    sin = sin.to(dtype=x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            if seq_len != 1:
                raise ValueError("cached decoding expects exactly one new token")
            k = torch.cat((past_kv[0], k), dim=2)
            v = torch.cat((past_kv[1], v), dim=2)

        present = (k, v)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=seq_len > 1,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y), present


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn(self.norm1(x), cos, sin, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                with torch.no_grad():
                    param.mul_(1.0 / math.sqrt(2 * cfg.n_layers))

    @property
    def head_dim(self) -> int:
        return self.cfg.d_model // self.cfg.n_heads

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        pos_offset: int = 0,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        batch, seq_len = idx.shape
        if pos_offset + seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence position {pos_offset + seq_len} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}"
            )

        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[pos_offset : pos_offset + seq_len]
        sin = self.rope_sin[pos_offset : pos_offset + seq_len]

        presents: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            past = past_kv[layer_idx] if past_kv is not None else None
            x, present = block(x, cos, sin, past)
            if presents is not None:
                presents.append(present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss, presents

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(param.numel() for param in self.parameters())
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            logits, _, past = self(idx, use_cache=True)
            for _ in range(max_new_tokens):
                step_logits = logits[:, -1, :]

                if temperature <= 0:
                    next_id = step_logits.argmax(dim=-1, keepdim=True)
                else:
                    step_logits = step_logits / temperature
                    if top_k is not None:
                        k = min(top_k, step_logits.size(-1))
                        values, _ = torch.topk(step_logits, k)
                        step_logits = step_logits.masked_fill(
                            step_logits < values[:, [-1]], float("-inf")
                        )
                    if top_p is not None and top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(
                            step_logits, descending=True, dim=-1
                        )
                        probs = F.softmax(sorted_logits, dim=-1)
                        cumulative = torch.cumsum(probs, dim=-1)
                        remove = cumulative > top_p
                        remove[..., 1:] = remove[..., :-1].clone()
                        remove[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        step_logits = torch.full_like(step_logits, float("-inf")).scatter(
                            1, sorted_idx, sorted_logits
                        )
                    probs = F.softmax(step_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, next_id), dim=1)
                if eos_id is not None and torch.all(next_id == eos_id):
                    break
                logits, _, past = self(
                    next_id,
                    past_kv=past,
                    pos_offset=idx.shape[1] - 1,
                    use_cache=True,
                )
        finally:
            self.train(was_training)
        return idx
