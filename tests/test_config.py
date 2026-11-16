from pathlib import Path

import pytest

from synax.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_tiny_config() -> None:
    cfg = load_config(ROOT / "configs" / "tiny.yaml")
    assert cfg.model.d_model == 128
    assert cfg.model.n_layers == 2
    assert cfg.model.n_heads == 4
    assert cfg.model.n_kv_heads == 2
    assert cfg.data.synthetic is True
    assert cfg.train.max_steps == 50


def test_roundtrip_dict() -> None:
    cfg = load_config(ROOT / "configs" / "tiny.yaml")
    rebuilt = Config.from_dict(cfg.to_dict())
    assert rebuilt.to_dict() == cfg.to_dict()


def test_unknown_key_rejected() -> None:
    cfg = load_config(ROOT / "configs" / "tiny.yaml")
    raw = cfg.to_dict()
    raw["model"]["not_a_real_option"] = 1
    with pytest.raises(ValueError, match="Unknown keys"):
        Config.from_dict(raw)


def test_invalid_head_split() -> None:
    cfg = load_config(ROOT / "configs" / "tiny.yaml")
    raw = cfg.to_dict()
    raw["model"]["n_heads"] = 5
    with pytest.raises(ValueError, match="divisible"):
        Config.from_dict(raw)
