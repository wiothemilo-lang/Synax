"""Evaluate a checkpoint on a validation shard."""

from __future__ import annotations

import argparse
import math

import torch

from synax.config import Config
from synax.data.dataset import TokenDataset
from synax.model.gpt import GPT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validation loss and perplexity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-bin", required=True)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])

    model = GPT(cfg.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    seq_len = args.seq_len or cfg.data.seq_len
    dataset = TokenDataset.from_file(args.val_bin, seq_len)

    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_enabled = True
    else:
        amp_dtype = torch.float32
        amp_enabled = False

    losses: list[float] = []
    with torch.no_grad():
        for index, (x, y) in enumerate(dataset.sequential_batches(args.batch_size)):
            if index >= args.max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                _, loss, _ = model(x, y)
            losses.append(float(loss))

    if not losses:
        raise SystemExit("validation shard produced no batches; lower batch size or seq len")

    mean_loss = sum(losses) / len(losses)
    print(f"checkpoint: {args.checkpoint}")
    print(f"step: {checkpoint['step']}")
    print(f"val_loss: {mean_loss:.4f}")
    print(f"val_ppl: {math.exp(min(mean_loss, 20)):.2f}")


if __name__ == "__main__":
    main()
