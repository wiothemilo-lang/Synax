"""Token shards: random-window batches for training, sequential batches for eval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    """A flat uint16 token array with random-window and sequential access.

    ``data`` can be a numpy array or a ``numpy.memmap``. Keeping it as a memmap
    means a 1B-token shard (2 GB) does not need to fit in RAM.
    """

    def __init__(self, data: np.ndarray, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len
        if len(data) <= seq_len + 1:
            raise ValueError(
                f"token shard has {len(data)} tokens, need more than {seq_len + 1}"
            )

    @classmethod
    def from_file(cls, path: str | Path, seq_len: int) -> TokenDataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"token shard not found: {path}")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        return cls(data, seq_len)

    def __len__(self) -> int:
        return max(0, (len(self.data) - 1) // self.seq_len)

    def random_batch(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high = len(self.data) - self.seq_len - 1
        if high <= 0:
            raise ValueError("token shard is too short for the requested seq_len")

        starts = torch.randint(high, (batch_size,), generator=generator).tolist()
        x = np.stack(
            [np.asarray(self.data[i : i + self.seq_len], dtype=np.int64) for i in starts]
        )
        y = np.stack(
            [np.asarray(self.data[i + 1 : i + 1 + self.seq_len], dtype=np.int64) for i in starts]
        )
        return torch.from_numpy(x), torch.from_numpy(y)

    def sequential_batches(
        self, batch_size: int
    ) -> "SequentialBatchIterator":
        return SequentialBatchIterator(self, batch_size)

    def token_count(self) -> int:
        return int(len(self.data))


class SequentialBatchIterator:
    """Iterate over non-overlapping windows, useful for stable validation loss."""

    def __init__(self, dataset: TokenDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        dataset = self.dataset
        n_windows = len(dataset)
        for start in range(0, n_windows - self.batch_size + 1, self.batch_size):
            starts = [
                (start + offset) * dataset.seq_len
                for offset in range(self.batch_size)
            ]
            x = np.stack(
                [
                    np.asarray(dataset.data[i : i + dataset.seq_len], dtype=np.int64)
                    for i in starts
                ]
            )
            y = np.stack(
                [
                    np.asarray(
                        dataset.data[i + 1 : i + 1 + dataset.seq_len], dtype=np.int64
                    )
                    for i in starts
                ]
            )
            yield torch.from_numpy(x), torch.from_numpy(y)


def synthetic_dataset(
    vocab_size: int,
    num_tokens: int,
    seq_len: int,
    seed: int = 0,
) -> TokenDataset:
    """Random tokens for smoke tests. Loss should hover around ln(vocab_size)."""

    rng = np.random.default_rng(seed)
    data = rng.integers(0, vocab_size, size=num_tokens, dtype=np.uint16)
    return TokenDataset(data, seq_len)
