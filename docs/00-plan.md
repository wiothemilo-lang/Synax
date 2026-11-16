# Kế hoạch Synax — train LLM tiếng Việt từ số 0 với 0đ

## Mục tiêu

Sở hữu toàn bộ pipeline: dữ liệu → tokenizer → kiến trúc → pretrain → SFT →
inference. Không API, không model có sẵn, không host ngoài.

Kết quả kỳ vọng: một model ~60M tham số nói tiếng Việt có ngữ pháp, làm được
một lĩnh vực hẹp. **Không** phải một trợ lý ngang ChatGPT.

## Quy mô và chi phí compute

Công thức: `FLOPs ≈ 6 × N × D`. Giả định GPU hiệu dụng 10–15 TFLOPS.

| Params | Token | Giờ GPU | Kết quả thực tế |
|---|---|---|---|
| 10M | 200M | ~0.3 | Học pipeline, chữ lộn xộn |
| 36M | 600M | ~3 | Câu có ngữ pháp, lặp ý |
| **60M** | **1B** | **~8–12** | **Nói được, viết đoạn ngắn** |
| 125M | 3B | ~50–60 | Ngang GPT-2, vẫn không phải trợ lý |
| 350M | 10B | ~12 ngày | Bắt đầu có ích ở lĩnh vực hẹp |

## Nơi chạy

| Nơi | Việc |
|---|---|
| Máy local (4 nhân, 7.7GB RAM) | Viết code, test, tokenizer nhỏ, inference Q4 |
| GitHub | Source, CI lint/test, Release model cuối |
| Colab Pro (L4/A100) | Train chính; dùng bf16 nếu GPU hỗ trợ |
| Kaggle (30h/tuần) | Dự phòng khi hết Colab compute units |
| Google Drive | Dataset tokenize + checkpoint |

Checkpoint mỗi 500 step, ghi thẳng Drive. Mọi session đều `--resume auto`.

## Kiến trúc

Llama-style decoder-only:

- RMSNorm, RoPE, SwiGLU, tied embeddings.
- GQA tùy chọn (`n_kv_heads < n_heads`).
- KV cache cho generation.
- 36M: `d_model=512`, 6 layer, 8 head, seq 512.
- 60M: `d_model=640`, 8 layer, 8 head, seq 1024, vocab 32k.

## Lịch 10 tuần

| Tuần | Việc | Milestone |
|---|---|---|
| 1 | Bootstrap repo, pipeline dữ liệu | Test xanh, chạy được trên 100MB |
| 2 | Lọc/dedupe 4–6GB, tokenizer 32k, encode 1B token | Round-trip tokenizer đạt, shard trên Drive |
| 3 | Train 10M trên 50M token | Overfit 1 batch, loss khởi tạo ≈ ln(32000) |
| 4 | Train 36M trên 600M token | Sinh câu tiếng Việt có ngữ pháp |
| 5–7 | Train 60M trên 1B token | Val loss giảm đều, sample đọc được |
| 8 | Chuẩn bị SFT, instruction tuning | Trả lời 10 lệnh đơn giản |
| 9 | Eval, quantize, GGUF, Ollama | Chat offline trên máy local |
| 10 | Báo cáo, loss curve, bài học | Repo tái lập được |

### Go/No-Go

- Hết tuần 4 mà sample vô nghĩa → dừng scale, sửa dữ liệu/tokenizer.
- Hết tuần 7 mà val loss không giảm → giảm model hoặc tăng dữ liệu.
- Không scale khi milestone trước chưa đạt.

## Hyperparameter pretrain

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.95), wd=0.1 |
| LR | 3e-4, cosine về 10%, warmup ~2% |
| Precision | bf16 nếu có, fp16 + GradScaler nếu không |
| Batch 60M | 8 seq × 1024 × grad_accum 4 = 32k token/step |
| Steps | ~31.000 cho 1B token |
| Grad clip | 1.0 |
| Checkpoint | Mỗi 500 step, giữ 2 bản |

## Đánh giá

- Val perplexity trên shard giữ riêng.
- 50 prompt định tính, chấm 1–5: trôi chảy, mạch lạc, đúng chủ đề.
- So sánh checkpoint 10k / 20k / 31k.
- Lĩnh vực hẹp: exact-match/F1 trên 100 câu tự soạn.

## Rủi ro

| Rủi ro | Xử lý |
|---|---|
| Dữ liệu rác | Đầu tư tuần 1–2, xem tay 100 mẫu |
| Colab ngắt | Checkpoint Drive + `--resume auto` |
| Tokenizer tệ với tiếng Việt | Test round-trip + tỷ lệ nén |
| Hết quota GPU | Kaggle 30h/tuần |
| Bản quyền dữ liệu | Wikipedia, public domain, giấy phép mở |
| Kỳ vọng sai | Đọc bảng quy mô ở trên |
