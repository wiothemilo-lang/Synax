"""Generate text from a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from synax.config import Config
from synax.model.gpt import GPT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sample from a trained Synax checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Việt Nam là")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])

    model = GPT(cfg.model).to(device)
    model.load_state_dict(checkpoint["model"])

    tokenizer_path = args.tokenizer or cfg.train.tokenizer_path
    if tokenizer_path and Path(tokenizer_path).exists():
        from synax.data.tokenizer import load_tokenizer

        sp = load_tokenizer(tokenizer_path)
        ids = sp.encode(args.prompt, out_type=int) or [sp.bos_id()]
        prompt_ids = ids
    else:
        sp = None
        prompt_ids = [1]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=sp.eos_id() if sp is not None else None,
    )

    if sp is not None:
        print(sp.decode(output[0].tolist()))
    else:
        print(output[0].tolist())


if __name__ == "__main__":
    main()
