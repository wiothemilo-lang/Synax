# Hướng dẫn chạy trên Google Colab

Hướng dẫn này dành cho notebook **`notebooks/colab_python.ipynb`** — train model
Python từ số 0.

## 0. Chuẩn bị

- Tài khoản Google + Colab Pro (bạn đã có).
- (Tùy chọn) Tài khoản Hugging Face để tải dataset nhanh hơn, tránh rate limit.
- Không cần cài gì trên máy local.

## 1. Mở notebook

Bấm link này để mở thẳng trong Colab:

```
https://colab.research.google.com/github/wiothemilo-lang/Synax/blob/main/notebooks/colab_python.ipynb
```

Hoặc:

1. Vào `https://colab.research.google.com`
2. `File → Open notebook`
3. Chọn tab `GitHub`
4. Dán `https://github.com/wiothemilo-lang/Synax`
5. Chọn `notebooks/colab_python.ipynb`

## 2. Chọn GPU

1. Menu `Runtime` → `Change runtime type`.
2. `Hardware accelerator` → chọn **GPU**.
3. Nếu có `GPU type`, chọn **L4** hoặc **A100**.
4. Bấm `Save`.

Cell số 2 sẽ in ra tên GPU. Nếu thấy `gpu: none` là chưa chọn đúng, làm lại bước 2.

## 3. Cách chạy cell

- Bấm vào cell, bấm nút ▶ bên trái, hoặc nhấn `Shift + Enter`.
- Chạy **tuần tự từ trên xuống**, không nhảy cell.
- Chờ cell chạy xong (dấu ✓ hoặc log dừng) mới chạy cell tiếp theo.

## 4. Từng cell

| Cell | Việc | Thời gian | Kết quả mong đợi |
|---|---|---|---|
| 1 | Markdown giới thiệu | — | — |
| 2 | Clone repo + cài package | 2–3 phút | `torch: ...`, `gpu: L4/A100`, `bf16: True` |
| 3 | Mount Google Drive | 10–30 giây | `drive: /content/drive/MyDrive/synax` |
| 4 | Markdown chọn chế độ | — | — |
| 5 | Cấu hình đường dẫn | 1 giây | `mode: test max_bytes: 200000000` |
| 6 | Markdown chuẩn bị corpus | — | — |
| 7 | Tải + lọc corpus 200MB | 5–10 phút | các dòng JSON `seen/kept/bytes` |
| 8 | Train tokenizer 32k | 5–10 phút | `tokenizer: .../sp32-code-test.model` |
| 9 | Encode shard | 2–5 phút | `train_tokens: ...`, `val_tokens: ...` |
| 10 | Markdown train | — | — |
| 11 | Train 200 step | 10–20 phút | `loss` giảm dần |
| 12 | Sample thử | 10–30 giây | đoạn code Python |

## 5. Hugging Face token (tùy chọn)

Làm nếu cell 7 báo rate limit hoặc tải chậm:

1. Vào `https://huggingface.co/settings/tokens`
2. `New token` → type `Read` → copy token.
3. Trong Colab, bấm biểu tượng 🔑 ở thanh bên trái (`Secrets`).
4. `Add new secret`:
   - Name: `HF_TOKEN`
   - Value: token vừa copy
5. Bật `Notebook access`.
6. Chạy lại cell 3.

Nếu không làm, cell 3 in `HF token: not set (anonymous access)` — vẫn chạy được.

## 6. Chạy test 200MB

Giữ nguyên trong cell 5:

```python
MODE = "test"
```

Sau đó chạy cell 7 → 8 → 9 → 11 → 12.

### Cell 7 — corpus

Log mẫu:

```json
{"seen": 20000, "license_skipped": 12100, "bad_code": 2100, "duplicate": 30, "kept": 5770, "bytes": 58200000}
```

Cuối cùng có một dòng stats đầy đủ. File nằm ở:

```
MyDrive/synax/python/test/clean/python.jsonl
```

### Cell 8 — tokenizer

Kết quả:

```
tokenizer: /content/drive/MyDrive/synax/tokenizer/sp32-code-test.model
```

### Cell 9 — encode

Kết quả:

```
train_tokens: 8,000,000
val_tokens: 40,000
```

### Cell 11 — train

Log mẫu:

```
device=cuda amp=torch.bfloat16 params=60,000,000 tokens/step=32,768 steps=200
{"step": 0, "lr": 1.5e-05, "loss": 10.37, "grad_norm": 1.2, ...}
{"step": 10, "lr": 1.5e-04, "loss": 8.9, ...}
{"step": 100, "lr": 3e-04, "loss": 6.5, ...}
{"step": 199, "loss": 5.8, ...}
```

Điều cần kiểm tra:

- `loss` bắt đầu quanh **10.3–10.4** (≈ ln 32.000).
- `loss` phải **giảm** trong 200 step.
- Nếu loss đứng yên quanh 10.37 → báo mình, đừng chạy full.

### Cell 12 — sample

Kết quả là một đoạn bắt đầu bằng `def fibonacci(n):`. Ở 200 step, code sẽ còn
lộn xộn nhưng phải có dấu hiệu của Python: `def`, `return`, dấu `:`, thụt lề.

## 7. Gửi lại kết quả

Copy và gửi mình:

1. 5–10 dòng log `loss` (step 0, 50, 100, 150, 199).
2. Dòng `train_tokens` / `val_tokens`.
3. Đoạn sample ở cell 12.

Mình sẽ xem loss có giảm đúng không và chỉnh hyperparameter trước khi chạy full.

## 8. Chạy full 4GB

Khi test ổn:

1. Sửa cell 5:

```python
MODE = "full"
```

2. Chạy lại cell 5 → 7 → 8 → 9 → 11 → 12.

| Bước | Thời gian |
|---|---|
| Corpus 4GB | 30–60 phút |
| Tokenizer 32k | 20–40 phút |
| Encode ~1B token | 20–40 phút |
| Train 31.000 step | 8–12 giờ, chia nhiều session |

Checkpoint nằm ở:

```
MyDrive/synax/checkpoints/60m-code-full/
```

Colab ngắt thì mở session mới, chạy cell 2 → 3 → 5, rồi chạy lại cell 11.
`--resume auto` sẽ đọc checkpoint mới nhất.

## 9. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: datasets` | Chạy lại cell 2 |
| `gpu: none` | Chọn lại GPU ở `Runtime → Change runtime type` |
| `CUDA out of memory` | Gửi mình log; sẽ giảm `batch_size` trong config |
| `Runtime disconnected` | Kết nối lại, chạy cell 2 → 3 → 5, rồi tiếp tục |
| Rate limit khi tải dataset | Thêm `HF_TOKEN` như mục 5 |
| `corpus exists` nhưng muốn làm lại | Xóa file trong `MyDrive/synax/python/<mode>/clean/` rồi chạy lại |
| `tokenizer exists` nhưng muốn làm lại | Xóa `sp32-code-<mode>.model` và `.vocab` trong Drive |
| `shards exist` nhưng muốn làm lại | Xóa `train.bin` và `val.bin` trong `.../tokenized/` |

## 10. Lưu ý

- Không đóng tab Colab khi đang train; Colab Pro có thể chạy nền một lúc nhưng
  vẫn có thể ngắt.
- Không xóa thư mục `MyDrive/synax` — đó là nơi chứa corpus và checkpoint.
- Không commit data/checkpoint vào git.
- 200MB chỉ để test; model thật cần ~4GB text (~1B token).
