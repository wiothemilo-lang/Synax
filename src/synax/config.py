"""Configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 640
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    ffn_hidden: int = 1760
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.n_kv_heads > self.n_heads:
            raise ValueError("n_kv_heads cannot exceed n_heads")


@dataclass
class TrainConfig:
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 600
    max_steps: int = 31000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    sample_every: int = 1000
    max_val_batches: int = 20
    keep_last: int = 2
    seed: int = 1337
    dtype: str = "auto"
    compile: bool = False
    tokenizer_path: str = ""
    sample_prompt: str = "Việt Nam là"


@dataclass
class DataConfig:
    train_bin: str = ""
    val_bin: str = ""
    seq_len: int = 1024
    synthetic: bool = False


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig
    out_dir: str = "checkpoints/run"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        allowed = {field.name for field in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown top-level keys: {sorted(unknown)}")
        return cls(
            model=_build(ModelConfig, raw.get("model")),
            train=_build(TrainConfig, raw.get("train")),
            data=_build(DataConfig, raw.get("data")),
            out_dir=raw.get("out_dir", "checkpoints/run"),
        )


def _build(cls: type, data: dict[str, Any] | None) -> Any:
    if data is None:
        return cls()
    allowed = {field.name for field in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a YAML mapping")
    return Config.from_dict(raw)
