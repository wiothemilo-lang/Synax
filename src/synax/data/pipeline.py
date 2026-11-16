"""Text cleaning, quality filtering and deduplication."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

_WS_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_TAG_RE = re.compile(r"<[^>]{1,200}>")


def normalize_text(text: str, strip_html: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    if strip_html:
        text = html.unescape(text)
        text = _TAG_RE.sub(" ", text)
    text = "".join(
        ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C")
    )
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def is_quality_text(
    text: str,
    min_chars: int = 200,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.5,
) -> bool:
    if not (min_chars <= len(text) <= max_chars):
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / len(text) < min_alpha_ratio:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(set(lines)) / len(lines) < 0.3:
        return False
    return True


def _iter_txt(path: Path) -> Iterator[str]:
    buffer: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                buffer.append(line.rstrip("\n"))
            elif buffer:
                yield "\n".join(buffer)
                buffer = []
    if buffer:
        yield "\n".join(buffer)


def iter_documents(paths: Iterable[str | Path]) -> Iterator[str]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content")
                    if text:
                        yield text
        elif path.suffix in {".txt", ".md"}:
            yield from _iter_txt(path)
        else:
            raise ValueError(f"unsupported input file: {path}")


def _shingles(text: str, size: int = 5) -> list[str]:
    words = text.split()
    if len(words) < size:
        return words
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


class MinHashDeduper:
    """Near-duplicate filter. Requires ``pip install datasketch``."""

    def __init__(self, threshold: float = 0.8, num_perm: int = 64) -> None:
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("install the optional dependency: pip install datasketch") from exc

        self._minhash_cls = MinHash
        self._num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._next_id = 0

    def add(self, text: str) -> bool:
        candidate = self._minhash_cls(num_perm=self._num_perm)
        for shingle in _shingles(text):
            candidate.update(shingle.encode("utf-8"))
        if self._lsh.query(candidate):
            return False
        self._lsh.insert(str(self._next_id), candidate)
        self._next_id += 1
        return True


def process_corpus(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    min_chars: int = 200,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.5,
    dedupe: str = "exact",
    minhash_threshold: float = 0.8,
    strip_html: bool = False,
) -> dict[str, int]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    minhash = MinHashDeduper(minhash_threshold) if dedupe == "minhash" else None

    stats = {"read": 0, "kept": 0, "short_or_bad": 0, "duplicate": 0}

    with output_path.open("w", encoding="utf-8") as handle:
        for raw_text in iter_documents(inputs):
            stats["read"] += 1
            text = normalize_text(raw_text, strip_html=strip_html)
            if not is_quality_text(
                text,
                min_chars=min_chars,
                max_chars=max_chars,
                min_alpha_ratio=min_alpha_ratio,
            ):
                stats["short_or_bad"] += 1
                continue

            if minhash is not None:
                if not minhash.add(text):
                    stats["duplicate"] += 1
                    continue
            elif dedupe == "exact":
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(digest)
            elif dedupe != "none":
                raise ValueError(f"unknown dedupe mode: {dedupe}")

            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean, filter and dedupe a text corpus")
    parser.add_argument("--input", nargs="+", required=True, help=".txt, .md or .jsonl files")
    parser.add_argument("--output", required=True, help="output .jsonl")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--min-alpha-ratio", type=float, default=0.5)
    parser.add_argument("--dedupe", choices=["none", "exact", "minhash"], default="exact")
    parser.add_argument("--minhash-threshold", type=float, default=0.8)
    parser.add_argument("--strip-html", action="store_true")
    args = parser.parse_args(argv)

    stats = process_corpus(
        args.input,
        args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        dedupe=args.dedupe,
        minhash_threshold=args.minhash_threshold,
        strip_html=args.strip_html,
    )
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
