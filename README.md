# Synax

Train một mô hình ngôn ngữ nhỏ **từ số 0**, với ngân sách **0đ**. Hỗ trợ hai
domain: văn bản tiếng Việt và code Python.

- Không gọi API của hãng AI nào.
- Không fine-tune model có sẵn: tự viết kiến trúc, tự train tokenizer, tự huấn luyện.
- Train trên GPU miễn phí (Colab Pro / Kaggle), inference chạy local.
- Data, checkpoint và weights **không bao giờ** commit vào git.

## Trạng thái

- [x] Bootstrap repo + CI
- [x] Kiến trúc GPT (RoPE, RMSNorm, SwiGLU, GQA, KV cache)
- [x] Vòng lặp pretrain: AMP, gradient accumulation, cosine LR, checkpoint/resume
- [x] Pipeline dữ liệu: chuẩn hoá NFC, lọc chất lượng, dedupe
- [x] Pipeline code Python: lọc license permissive, kiểm tra AST, dedupe
- [x] Tokenizer SentencePiece + encode ra shard `uint16`, chế độ code giữ indentation
- [x] Test + GitHub Actions
- [ ] Dataset 1B token (tuần 2)
- [ ] Pretrain 60M (tuần 5–7)
- [ ] SFT (tuần 8)
- [ ] Convert GGUF + chạy Ollama (tuần 9)

Kế hoạch đầy đủ: [`docs/00-plan.md`](docs/00-plan.md).
Dữ liệu văn bản: [`docs/01-data.md`](docs/01-data.md).
Code Python: [`docs/02-python.md`](docs/02-python.md).

## Quick start (local, CPU)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev]"

.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python -m synax.train.pretrain --config configs/tiny.yaml --synthetic --steps 50
```

Lệnh cuối chạy 50 step trên dữ liệu ngẫu nhiên để kiểm tra toàn bộ pipeline
(model, optimizer, checkpoint, log). Nó không tạo ra model có ích.

> Dung lượng local: `.venv` với torch CPU khoảng **1.1 GB**. Nếu chỉ làm việc
> trên Colab, bạn không cần cài local; clone repo chỉ vài trăm KB.

## Colab Pro

- Văn bản tiếng Việt: [`notebooks/colab_pretrain.ipynb`](notebooks/colab_pretrain.ipynb)
- Code Python: [`notebooks/colab_python.ipynb`](notebooks/colab_python.ipynb)

Chọn GPU (L4/A100 nếu có), mount Google Drive và chạy lần lượt các cell.
Notebook code Python chạy thử 200MB trước, sau đó mới chạy full 4GB.

Hướng dẫn từng bước + xử lý lỗi: [`docs/03-colab.md`](docs/03-colab.md).

## Domain code Python

```bash
pip install -e ".[data]"

python -m synax.data.code --output data/clean/python.jsonl --max-bytes 4000000000
python -m synax.data.tokenizer train \
  --input data/clean/python.jsonl --output-prefix tokenizer/sp32-code \
  --vocab-size 32000 --code
python -m synax.data.tokenizer encode \
  --input data/clean/python.jsonl --tokenizer tokenizer/sp32-code.model \
  --output data/tokenized/python-train.bin \
  --val-output data/tokenized/python-val.bin --val-ratio 0.005 --code
python -m synax.train.pretrain --config configs/model-60m-code.yaml
```

Chi tiết dataset, giấy phép và thời gian: [`docs/02-python.md`](docs/02-python.md).

## Cấu trúc

```
configs/            tiny (smoke), 36M, 60M, 60M-code
src/synax/
  config.py         dataclass + YAML
  model/gpt.py      Llama-style decoder
  data/             pipeline văn bản, code Python, tokenizer, dataset
  train/pretrain.py training loop + resume
  eval/evaluate.py  val loss / perplexity
  inference/        sampling
scripts/            download, release, smoke test
notebooks/          Colab (văn bản + code Python)
tests/              pytest
docs/               plan, data, python
```

## Quy tắc artifact

| Thứ | Nơi lưu | Vào git? |
|---|---|---|
| Source, config, docs | GitHub | ✅ |
| Dataset thô / đã tokenize | Drive / Kaggle Dataset | ❌ |
| Checkpoint | Drive | ❌ |
| Model cuối (GGUF) | GitHub Release | ❌ |

`.gitignore` đã chặn `data/`, `checkpoints/`, `*.bin`, `*.pt`, `*.gguf`,
`*.safetensors`. Đừng dùng `git add -f` cho các file này.

## License

MIT cho code. Dữ liệu huấn luyện giữ giấy phép gốc của nó.
