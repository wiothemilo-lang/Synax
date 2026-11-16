"""SentencePiece training and corpus encoding into uint16 shards."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from synax.data.pipeline import iter_documents, normalize_text


def train_tokenizer(
    inputs: list[str | Path],
    output_prefix: str | Path,
    *,
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    strip_html: bool = False,
    tmp_dir: str | Path | None = None,
) -> Path:
    """Train a SentencePiece model on .txt/.md/.jsonl inputs and return the .model path."""

    import sentencepiece as spm

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".txt",
        delete=False,
        dir=str(tmp_dir) if tmp_dir else None,
    ) as handle:
        corpus_path = Path(handle.name)
        for text in iter_documents(inputs):
            text = normalize_text(text, strip_html=strip_html)
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                while len(line) > 4000:
                    cut = line.rfind(" ", 0, 4000)
                    if cut <= 0:
                        cut = 4000
                    handle.write(line[:cut] + "\n")
                    line = line[cut:].lstrip()
                handle.write(line + "\n")

    try:
        spm.SentencePieceTrainer.train(
            input=str(corpus_path),
            model_prefix=str(output_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            normalization_rule_name="identity",
            byte_fallback=True,
            split_by_unicode_script=True,
            split_digits=True,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            input_sentence_size=5_000_000,
            shuffle_input_sentence=True,
            max_sentence_length=16384,
            num_threads=max(1, (os.cpu_count() or 4) - 1),
            hard_vocab_limit=False,
        )
    finally:
        corpus_path.unlink(missing_ok=True)

    return output_prefix.with_suffix(".model")


def load_tokenizer(model_path: str | Path):
    import sentencepiece as spm

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer not found: {path}")
    return spm.SentencePieceProcessor(model_file=str(path))


def _is_validation(text: str, ratio: float) -> bool:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32 < ratio


def encode_corpus(
    inputs: list[str | Path],
    tokenizer_path: str | Path,
    output: str | Path,
    *,
    val_output: str | Path | None = None,
    val_ratio: float = 0.0,
    add_eos: bool = True,
    strip_html: bool = False,
) -> tuple[int, int]:
    """Encode documents into flat uint16 shards. Returns (train_tokens, val_tokens)."""

    sp = load_tokenizer(tokenizer_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_path = Path(val_output) if val_output else None
    if val_path is not None:
        val_path.parent.mkdir(parents=True, exist_ok=True)

    train_tokens = 0
    val_tokens = 0
    val_context = val_path.open("wb") if val_path is not None else nullcontext()

    with output_path.open("wb") as train_handle, val_context as val_handle:
        for raw_text in iter_documents(inputs):
            text = normalize_text(raw_text, strip_html=strip_html)
            if not text:
                continue

            ids = sp.encode(text, out_type=int)
            if add_eos:
                ids.append(sp.eos_id())

            chunk = np.asarray(ids, dtype=np.uint16)
            if val_handle is not None and val_ratio > 0 and _is_validation(text, val_ratio):
                val_handle.write(chunk.tobytes())
                val_tokens += len(ids)
            else:
                train_handle.write(chunk.tobytes())
                train_tokens += len(ids)

    return train_tokens, val_tokens


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a tokenizer or encode a corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train SentencePiece")
    train_parser.add_argument("--input", nargs="+", required=True)
    train_parser.add_argument("--output-prefix", required=True)
    train_parser.add_argument("--vocab-size", type=int, default=32000)
    train_parser.add_argument("--model-type", default="bpe", choices=["bpe", "unigram"])
    train_parser.add_argument("--character-coverage", type=float, default=0.9995)
    train_parser.add_argument("--strip-html", action="store_true")
    train_parser.add_argument("--tmp-dir", default=None)

    encode_parser = subparsers.add_parser("encode", help="encode corpus to uint16")
    encode_parser.add_argument("--input", nargs="+", required=True)
    encode_parser.add_argument("--tokenizer", required=True)
    encode_parser.add_argument("--output", required=True)
    encode_parser.add_argument("--val-output", default=None)
    encode_parser.add_argument("--val-ratio", type=float, default=0.0)
    encode_parser.add_argument("--no-eos", action="store_true")
    encode_parser.add_argument("--strip-html", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "train":
        model_path = train_tokenizer(
            args.input,
            args.output_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            strip_html=args.strip_html,
            tmp_dir=args.tmp_dir,
        )
        print(f"tokenizer: {model_path}")
    else:
        train_tokens, val_tokens = encode_corpus(
            args.input,
            args.tokenizer,
            args.output,
            val_output=args.val_output,
            val_ratio=args.val_ratio,
            add_eos=not args.no_eos,
            strip_html=args.strip_html,
        )
        print(f"train_tokens: {train_tokens:,}")
        if args.val_output:
            print(f"val_tokens: {val_tokens:,}")


if __name__ == "__main__":
    main()
