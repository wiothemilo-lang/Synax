import numpy as np
import torch

from synax.data.dataset import TokenDataset, synthetic_dataset


def write_shard(path, tokens: np.ndarray) -> None:
    path.write_bytes(tokens.astype(np.uint16).tobytes())


def test_random_batch_shapes(tmp_path) -> None:
    tokens = np.arange(2000, dtype=np.uint16) % 512
    path = tmp_path / "train.bin"
    write_shard(path, tokens)

    dataset = TokenDataset.from_file(path, seq_len=32)
    x, y = dataset.random_batch(4, generator=torch.Generator().manual_seed(0))

    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert x.dtype == torch.long


def test_sequential_batches(tmp_path) -> None:
    tokens = np.arange(1024, dtype=np.uint16)
    path = tmp_path / "val.bin"
    write_shard(path, tokens)

    dataset = TokenDataset.from_file(path, seq_len=16)
    batches = list(dataset.sequential_batches(batch_size=4))

    assert batches
    x, y = batches[0]
    assert x.shape == (4, 16)
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_synthetic_dataset() -> None:
    dataset = synthetic_dataset(vocab_size=64, num_tokens=10_000, seq_len=16, seed=0)
    x, y = dataset.random_batch(2)
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert int(x.max()) < 64
    assert dataset.token_count() == 10_000


def test_too_short_shard_rejected(tmp_path) -> None:
    path = tmp_path / "short.bin"
    write_shard(path, np.arange(8, dtype=np.uint16))
    try:
        TokenDataset.from_file(path, seq_len=32)
    except ValueError as exc:
        assert "need more than" in str(exc)
    else:
        raise AssertionError("expected ValueError")
