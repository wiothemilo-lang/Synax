# Dữ liệu

## Nguồn đề xuất

| Lĩnh vực | Nguồn | Ghi chú |
|---|---|---|
| Tổng quát tiếng Việt | Wikipedia vi dump, CC-100 vi, Oscar vi | Kiểm tra điều khoản từng nguồn |
| Sách | Project Gutenberg (public domain) | Ít bản tiếng Việt |
| Code | The Stack v2 (gated), repo giấy phép mở | Cần đồng ý điều khoản |
| Hẹp | Văn bản pháp luật, y khoa, truyện | Kiểm tra quyền sử dụng |

Model đầu tiên nên chọn **một lĩnh vực hẹp** — 60M tham số cho kết quả tốt hơn
nhiều so với "tổng quát".

## Quy trình

```
raw .txt/.jsonl
   │  synax.data.pipeline
   ▼
clean .jsonl          # NFC, lọc chất lượng, dedupe
   │  synax.data.tokenizer train
   ▼
sp32.model            # SentencePiece BPE 32k
   │  synax.data.tokenizer encode
   ▼
train.bin / val.bin   # uint16, 1 token = 2 byte
```

## Lệnh

```bash
# 1. Làm sạch + dedupe
python -m synax.data.pipeline \
  --input raw/*.txt \
  --output data/clean/corpus.jsonl \
  --min-chars 200 --max-chars 200000 \
  --dedupe exact

# 2. Train tokenizer 32k
python -m synax.data.tokenizer train \
  --input data/clean/corpus.jsonl \
  --output-prefix tokenizer/sp32 \
  --vocab-size 32000 \
  --tmp-dir data/tmp

# 3. Encode thành shard
python -m synax.data.tokenizer encode \
  --input data/clean/corpus.jsonl \
  --tokenizer tokenizer/sp32.model \
  --output data/tokenized/train.bin \
  --val-output data/tokenized/val.bin \
  --val-ratio 0.005
```

## Kiểm tra tokenizer

- `decode(encode(text)) == text` với 1.000 câu tiếng Việt có đủ dấu.
- Tỷ lệ nén nên đạt **3–4 ký tự/token**. Dưới 2.5 → vocab chưa đủ tốt.
- Token id phải nằm trong `uint16` (< 65.536). Vocab 32k an toàn.

## Dung lượng

| Thứ | Kích thước |
|---|---|
| 1B token | 2 GB (`uint16`) |
| Corpus thô cho 1B token | 4–6 GB |
| Checkpoint 60M (model + optimizer) | ~0.7–1 GB |
| Model 60M Q4 | ~40 MB |

## Lưu trữ

- Corpus sạch và shard: Google Drive `MyDrive/synax/` hoặc Kaggle Dataset.
- Không commit `data/` — `.gitignore` đã chặn.
- Xoá checkpoint cũ sau khi đã chọn bản tốt nhất.

## Lưu ý pháp lý

- Wikipedia: CC BY-SA, được dùng để train, cần ghi nguồn nếu phân phối lại.
- Code GitHub: chỉ dùng repo có giấy phép cho phép; The Stack có điều khoản riêng.
- Không crawl dữ liệu vi phạm robots.txt hoặc điều khoản dịch vụ.
