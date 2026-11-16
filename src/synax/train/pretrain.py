"""Pretraining loop with AMP, gradient accumulation, checkpointing and resume."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from synax.config import Config, TrainConfig, load_config
from synax.data.dataset import TokenDataset, synthetic_dataset
from synax.model.gpt import GPT


def get_lr(step: int, cfg: TrainConfig) -> float:
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - min_lr)


def build_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


def resolve_amp(cfg: TrainConfig, device: str) -> tuple[torch.dtype, bool, bool]:
    if device != "cuda":
        return torch.float32, False, False

    if cfg.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif cfg.dtype == "bf16":
        dtype = torch.bfloat16
    elif cfg.dtype == "fp16":
        dtype = torch.float16
    elif cfg.dtype == "fp32":
        dtype = torch.float32
    else:
        raise ValueError(f"unknown dtype: {cfg.dtype}")

    enabled = dtype != torch.float32
    return dtype, enabled, dtype == torch.float16


def make_datasets(cfg: Config) -> tuple[TokenDataset, TokenDataset]:
    if cfg.data.synthetic:
        train_tokens = max(cfg.data.seq_len * 8192, 100_000)
        train_ds = synthetic_dataset(
            cfg.model.vocab_size,
            train_tokens,
            cfg.data.seq_len,
            cfg.train.seed,
        )
        val_ds = synthetic_dataset(
            cfg.model.vocab_size,
            train_tokens // 10,
            cfg.data.seq_len,
            cfg.train.seed + 1,
        )
        return train_ds, val_ds

    train_ds = TokenDataset.from_file(cfg.data.train_bin, cfg.data.seq_len)
    val_ds = TokenDataset.from_file(cfg.data.val_bin, cfg.data.seq_len)
    return train_ds, val_ds


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: TokenDataset,
    cfg: Config,
    device: str,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    for index, (x, y) in enumerate(dataset.sequential_batches(cfg.train.batch_size)):
        if index >= cfg.train.max_val_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
            _, loss, _ = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def save_checkpoint(
    out_dir: str | Path,
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: Config,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ckpt_{step:07d}.pt"
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg.to_dict(),
    }
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    (out_dir / "latest.txt").write_text(path.name, encoding="utf-8")

    keep_last = cfg.train.keep_last
    if keep_last > 0:
        checkpoints = sorted(out_dir.glob("ckpt_*.pt"))
        for old in checkpoints[:-keep_last]:
            old.unlink(missing_ok=True)
    return path


def load_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str = "cpu",
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def sample_text(model: GPT, cfg: Config, device: str, max_new_tokens: int = 80) -> str | None:
    tokenizer_path = cfg.train.tokenizer_path
    if not tokenizer_path or not Path(tokenizer_path).exists():
        return None

    from synax.data.tokenizer import load_tokenizer

    sp = load_tokenizer(tokenizer_path)
    ids = sp.encode(cfg.train.sample_prompt, out_type=int)
    if not ids:
        ids = [sp.bos_id()]
    ids = ids[: max(1, cfg.model.max_seq_len - max_new_tokens)]

    x = torch.tensor([ids], dtype=torch.long, device=device)
    output = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        eos_id=sp.eos_id(),
    )
    return sp.decode(output[0].tolist())


def append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pretrain Synax from scratch")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint path or 'auto'")
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--synthetic", action="store_true", help="random tokens, for smoke tests")
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-bin", default=None, help="override data.train_bin")
    parser.add_argument("--val-bin", default=None, help="override data.val_bin")
    parser.add_argument("--out-dir", default=None, help="override out_dir")
    parser.add_argument("--tokenizer", default=None, help="override train.tokenizer_path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.train.max_steps = args.steps
    if args.synthetic:
        cfg.data.synthetic = True
    elif args.train_bin is not None:
        cfg.data.synthetic = False
    if args.train_bin is not None:
        cfg.data.train_bin = args.train_bin
    if args.val_bin is not None:
        cfg.data.val_bin = args.val_bin
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.tokenizer is not None:
        cfg.train.tokenizer_path = args.tokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)

    train_ds, val_ds = make_datasets(cfg)
    model = GPT(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg.train)
    amp_dtype, amp_enabled, use_fp16 = resolve_amp(cfg.train, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if use_fp16 else None

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"

    start_step = 0
    if args.resume:
        if args.resume == "auto":
            latest = out_dir / "latest.txt"
            if latest.exists():
                resume_path = out_dir / latest.read_text(encoding="utf-8").strip()
            else:
                resume_path = None
        else:
            resume_path = Path(args.resume)

        if resume_path is not None:
            checkpoint = load_checkpoint(resume_path, model, optimizer, scaler, device)
            start_step = int(checkpoint["step"]) + 1
            print(f"resumed {resume_path} at step {start_step}")
        else:
            print("no checkpoint found, starting fresh")

    generator = torch.Generator().manual_seed(cfg.train.seed + start_step)
    tokens_per_step = cfg.train.batch_size * cfg.train.grad_accum * cfg.data.seq_len
    print(
        f"device={device} amp={amp_dtype} params={model.num_parameters():,} "
        f"tokens/step={tokens_per_step:,} steps={cfg.train.max_steps:,}"
    )

    started = time.time()
    model.train()
    for step in range(start_step, cfg.train.max_steps):
        lr = get_lr(step, cfg.train)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.train.grad_accum):
            x, y = train_ds.random_batch(cfg.train.batch_size, generator=generator)
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                _, loss, _ = model(x, y)
            scaled_loss = loss / cfg.train.grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            loss_accum += scaled_loss.detach().item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % cfg.train.log_every == 0:
            record = {
                "step": step,
                "lr": lr,
                "loss": loss_accum,
                "grad_norm": float(grad_norm),
                "tokens": (step + 1) * tokens_per_step,
                "elapsed_s": round(time.time() - started, 1),
            }
            print(json.dumps(record))
            append_log(log_path, record)

        if cfg.train.eval_every > 0 and (step + 1) % cfg.train.eval_every == 0:
            val_loss = evaluate(model, val_ds, cfg, device, amp_dtype, amp_enabled)
            record = {
                "step": step,
                "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20)),
            }
            print(json.dumps(record))
            append_log(log_path, record)

        if cfg.train.sample_every > 0 and (step + 1) % cfg.train.sample_every == 0:
            text = sample_text(model, cfg, device)
            if text:
                print(f"--- sample @ step {step} ---")
                print(text)
                print("---")

        if cfg.train.save_every > 0 and (step + 1) % cfg.train.save_every == 0:
            path = save_checkpoint(out_dir, step, model, optimizer, scaler, cfg)
            print(f"saved {path}")

    final_path = save_checkpoint(out_dir, cfg.train.max_steps - 1, model, optimizer, scaler, cfg)
    print(f"done: {final_path}")


if __name__ == "__main__":
    main()
