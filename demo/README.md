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
   ├─ 1. Lập kế hoạch          lấy mẫu n kế hoạch tính toán khác nhau, T = 0.6
   └─ 2. Bỏ phiếu & thực thi   chuẩn hoá, gom cụm, cụm đông nhất thì đem chạy
   │
   └─▶  subtract(7.50, 6.75)  →  0.75
```

Ý chính: **không tin một đường suy luận duy nhất** — lấy nhiều kế hoạch độc lập
rồi cho chúng bỏ phiếu, thay vì giải mã tham lam một lần.

## Khác paper ở đâu

Đây là cài đặt của Nguyen, Ha, Le & Vu, *"A Graph-Based Agent Approach to Numerical
Reasoning Question Answering"* (VLSP 2025) — hệ thắng Subtask 2 với EA 84.00%.

Paper có **4 node**. Bản demo này chạy **2**: hai node phân rã câu hỏi (§4.1 Subquery
Generator, §4.2 Subquery Answerer) đã tắt qua `use_decomposition=False`. Chúng vẫn nằm
trong đồ thị nhưng no-op — không gọi API, không hiện lên giao diện.

Đó đúng là dòng **"Multi-path only"** trong bảng ablation của paper, và theo số liệu
của chính họ thì phần bỏ đi rất rẻ:

| Cấu hình | 8B EA / PA | 32B EA / PA |
|---|---|---|
| Đầy đủ 4 node | 78.47 / 71.83 | 81.29 / 75.25 |
| **Multi-path only (bản demo này)** | **78.38 / 70.91** | **80.91 / 74.65** |
| Decomposition only (n=1) | 72.84 / 62.58 | 79.48 / 66.80 |
| Nhắc thẳng, không pipeline | 41.05 / 32.60 | 47.89 / 40.24 |

Mất ~0,1–0,4 EA, đổi lại nhanh hơn nhiều lần. Bỏ đa đường mới là thứ đắt: mất 5,6 EA.

Các tham số còn lại giữ nguyên của paper: `n=15`, `T=0.6`, `top_p=0.95`, `top_k=20`,
prompt tiếng Việt, bỏ phiếu canonical. `server.py` gọi thẳng `agentic/`, không sửa dòng nào.

> Hạ `n` xuống cho chạy nhanh là đang bớt nốt phần đa đường — thứ mang gần như
> toàn bộ điểm số. Lúc đo đạc thì để `n = 15`.

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

Cùng một câu hỏi, trước và sau khi tắt phân rã (DeepSeek-V4-Flash):

| | Thời gian | Lời gọi | Token |
|---|---|---|---|
| 4 node, n=5 | 73,5 s | 7 | 13.271 |
| **2 node, n=5** | **12,2 s** | **2** | **4.787** |
| 4 node, n=15 (PMI) | 55,0 s | 7 | 14.535 |
| **2 node, n=15 (PMI)** | **4,8 s** | **2** | **5.744** |

Đáp án không đổi trong cả bốn lượt.

Ba model, cùng câu hỏi, n=5, đều ra `subtract(7.50, 6.75)` → `0.75` (khớp `exe_ans`):

| Model | Thời gian |
|---|---|
| gemma-4-31B-it | ~2 s |
| gpt-oss-120b | ~4 s |
| DeepSeek-V4-Flash | 12,2 s |

Model reasoning chậm hơn vì nghĩ lâu trước khi viết kế hoạch.
