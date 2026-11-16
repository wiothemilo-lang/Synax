# Synax

Train một mô hình ngôn ngữ tiếng Việt **từ số 0**, với ngân sách **0đ**.

- Không gọi API của hãng AI nào.
- Không fine-tune model có sẵn: tự viết kiến trúc, tự train tokenizer, tự huấn luyện.
- Train trên GPU miễn phí (Colab Pro / Kaggle), inference chạy local.
- Data, checkpoint và weights **không bao giờ** commit vào git.

## Trạng thái

- [x] Bootstrap repo + CI
- [x] Kiến trúc GPT (RoPE, RMSNorm, SwiGLU, GQA, KV cache)
- [x] Vòng lặp pretrain: AMP, gradient accumulation, cosine LR, checkpoint/resume
- [x] Pipeline dữ liệu: chuẩn hoá NFC, lọc chất lượng, dedupe
- [x] Tokenizer SentencePiece + encode ra shard `uint16`
- [x] Test + GitHub Actions
- [ ] Dataset 1B token (tuần 2)
- [ ] Pretrain 60M (tuần 5–7)
- [ ] SFT (tuần 8)
- [ ] Convert GGUF + chạy Ollama (tuần 9)

Kế hoạch đầy đủ: [`docs/00-plan.md`](docs/00-plan.md). Pipeline dữ liệu: [`docs/01-data.md`](docs/01-data.md).

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

Mở [`notebooks/colab_pretrain.ipynb`](notebooks/colab_pretrain.ipynb), chọn GPU
(L4/A100 nếu có), mount Google Drive và chạy lần lượt các cell.

Notebook sẽ:

1. Clone repo này.
2. Train tokenizer 32k trên corpus của bạn.
3. Encode corpus thành shard `uint16`.
4. Pretrain với checkpoint ghi thẳng vào Drive.
5. Sample thử từ checkpoint mới nhất.

## Cấu trúc

```
configs/            tiny (smoke test), 36M, 60M
src/synax/
  config.py         dataclass + YAML
  model/gpt.py      Llama-style decoder
  data/             pipeline, tokenizer, dataset
  train/pretrain.py training loop + resume
  eval/evaluate.py  val loss / perplexity
  inference/        sampling
scripts/            download, release, smoke test
notebooks/          Colab
tests/              pytest
docs/               plan + data notes
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
