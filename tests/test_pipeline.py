import json
import unicodedata

from synax.data.pipeline import (
    is_quality_text,
    normalize_text,
    process_corpus,
)


def test_normalize_text_is_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "Việt Nam")
    normalized = normalize_text(decomposed)
    assert normalized == "Việt Nam"
    assert unicodedata.is_normalized("NFC", normalized)


def test_normalize_text_collapses_whitespace() -> None:
    text = "  Xin   chào\t\tthế giới \n\n\n\nHẹn gặp lại  "
    assert normalize_text(text) == "Xin chào thế giới\n\nHẹn gặp lại"


def test_quality_filter() -> None:
    good = "Đây là một đoạn văn tiếng Việt đủ dài để vượt qua bộ lọc chất lượng. " * 10
    assert is_quality_text(good, min_chars=100)
    assert not is_quality_text("quá ngắn", min_chars=100)
    assert not is_quality_text("12345 " * 100, min_chars=100)


def test_exact_dedupe(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    paragraph = "Một đoạn văn tiếng Việt đủ dài để được giữ lại trong tập dữ liệu. " * 8
    corpus.write_text(
        f"{paragraph}\n\n{paragraph}\n\n{paragraph} khác một chút.\n",
        encoding="utf-8",
    )

    output = tmp_path / "clean.jsonl"
    stats = process_corpus([corpus], output, min_chars=100, dedupe="exact")

    assert stats["read"] == 3
    assert stats["kept"] == 2
    assert stats["duplicate"] == 1

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all("text" in row for row in rows)
