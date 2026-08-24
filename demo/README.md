# Demo — MPR-Agent

Giao diện chat hỏi số liệu trên báo cáo tài chính tiếng Việt. Gõ câu hỏi, hệ thống
chạy 4 tác tử rồi trả về **phương trình** và **đáp án**, hiện rõ từng bước nó nghĩ.

## Chạy

```bash
pip install -r requirements.txt
python server.py            # http://localhost:8000
```

Cần `API_KEY` và `BASE_URL` trong `.env` (xem `.env.example`). Khoá nằm ở server,
không ra tới trình duyệt. Không có backend thì giao diện vẫn mở được nhưng chỉ chạy mô phỏng.

## Dùng thế nào

Đầu vào có 3 phần, đúng như một dòng trong tập ViNumQA:

| Phần | Nhập ở đâu |
|---|---|
| **Ngữ cảnh** — đoạn văn quanh bảng | nút `+` → *Ngữ cảnh* |
| **Bảng** — số liệu | nút `+` → *Bảng dữ liệu* (dán thẳng từ Excel được) |
| **Câu hỏi** | gõ vào khung chat |

Lười nhập thì bấm một thẻ ví dụ ở dưới — ví dụ rút thẳng từ `datasets/ViNumQA/origin/train.json`,
bấm *Ví dụ khác* để đổi bộ mới.

Trên header đổi được **model** (`DeepSeek-V4-Flash`, `gemma-4-31B-it`, `gpt-oss-120b`)
và **n** — số kế hoạch lấy mẫu.

## Nó làm gì

```
Câu hỏi + tài liệu
   │
   ├─ 1. Tách câu hỏi con        "giá PMI 2012 là bao nhiêu?" · chỉ tra số, không tính
   ├─ 2. Trả lời câu hỏi con      mỗi câu một lời gọi, chạy song song
   ├─ 3. Lập kế hoạch             lấy mẫu n kế hoạch tính toán khác nhau
   └─ 4. Bỏ phiếu & thực thi      kế hoạch nào nhiều phiếu nhất thì chạy
   │
   └─▶  subtract(7.50, 6.75)  →  0.75
```

Ý chính: **tách việc tìm số ra khỏi việc làm toán**, và **không tin một đường suy luận
duy nhất** — lấy nhiều kế hoạch rồi cho chúng bỏ phiếu.

Đây là cài đặt của Nguyen, Ha, Le & Vu, *"A Graph-Based Agent Approach to Numerical
Reasoning Question Answering"* (VLSP 2025) — hệ thắng Subtask 2 với EA 84.00%.
`server.py` gọi thẳng `agentic/`, giữ nguyên tham số của paper: `n=15`, `T=0.6`,
`top_p=0.95`, `top_k=20`, prompt tiếng Việt, bỏ phiếu canonical.

> Hạ `n` xuống cho chạy nhanh thì đang bớt phần đa đường — thành cấu hình
> *"Decomposition only"* trong bảng ablation, không còn là hệ đầy đủ.

## File

| File | |
|---|---|
| `server.py` | HTTP + SSE, chạy pipeline thật |
| `index.html` `styles.css` `app.js` | giao diện |
| `agentic/` `scorer.py` | pipeline + bộ chấm — **bản sao nguyên văn**, sửa ở bản gốc trong `notebooks/` |
| `make_examples.py` → `examples.json` | rút ví dụ từ tập train |
| `build.py` → `preview.html` | gộp thành 1 file tự chứa để gửi đi xem |

Thư mục này chạy độc lập được, không cần phần còn lại của repo — nhớ mang theo `.env`.

## Đo thật

| Model | n | Thời gian | Kết quả |
|---|---|---|---|
| gemma-4-31B-it | 3 | 2,7 s | `subtract(7.50, 6.75)` → `0.75` ✓ |
| gpt-oss-120b | 3 | 5,9 s | `subtract(7.50, 6.75)` → `0.75` ✓ |
| DeepSeek-V4-Flash | 3 | 28,5 s | `subtract(7.50, 6.75)` → `0.75` ✓ |
| DeepSeek-V4-Flash | 15 | 55 s | `subtract(108.50, 100.00), divide(#0, 100.00)` → `0.085` ✓ |

Model reasoning chậm hơn nhiều vì chặng tách câu hỏi con phải nghĩ lâu.
