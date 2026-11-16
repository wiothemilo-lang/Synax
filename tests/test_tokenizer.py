import importlib.util

import numpy as np
import pytest

from synax.data.tokenizer import encode_corpus, load_tokenizer, train_tokenizer

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sentencepiece") is None,
    reason="sentencepiece is not installed",
)


SENTENCES = [
    "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương.",
    "Hà Nội là thủ đô và là thành phố lớn thứ hai của Việt Nam.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế lớn nhất cả nước.",
    "Học máy là một lĩnh vực của trí tuệ nhân tạo.",
    "Mô hình ngôn ngữ dự đoán token tiếp theo trong một chuỗi.",
    "Dữ liệu sạch quan trọng hơn số lượng dữ liệu.",
    "Chúng ta cần kiểm tra chất lượng trước khi huấn luyện.",
    "Một tokenizer tốt giúp tiếng Việt được mã hoá hiệu quả.",
]


def make_corpus(path) -> None:
    blocks = []
    for i in range(40):
        block = " ".join(f"{sentence} Đây là câu thứ {i} trong tập mẫu." for sentence in SENTENCES)
        blocks.append(block)
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def test_train_and_roundtrip(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    make_corpus(corpus)

    model_path = train_tokenizer(
        [corpus],
        tmp_path / "sp",
        vocab_size=512,
        character_coverage=1.0,
    )
    assert model_path.exists()

    sp = load_tokenizer(model_path)
    text = "Việt Nam là một quốc gia xinh đẹp."
    ids = sp.encode(text, out_type=int)
    assert ids
    assert sp.decode(ids) == text


def test_encode_corpus_creates_shards(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    make_corpus(corpus)

    model_path = train_tokenizer([corpus], tmp_path / "sp", vocab_size=512)
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"

    train_tokens, val_tokens = encode_corpus(
        [corpus],
        model_path,
        train_bin,
        val_output=val_bin,
        val_ratio=0.2,
    )

    assert train_tokens > 0
    assert val_tokens > 0
    assert train_bin.stat().st_size == train_tokens * 2
    assert val_bin.stat().st_size == val_tokens * 2

    data = np.memmap(train_bin, dtype=np.uint16, mode="r")
    assert len(data) == train_tokens
