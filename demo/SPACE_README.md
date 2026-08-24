---
title: Num Reasoning for Vietnamese Financial Text
emoji: 🧮
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Hỏi đáp số liệu trên báo cáo tài chính tiếng Việt bằng hệ đa tác tử
---

# Num Reasoning for Vietnamese Financial Text

Hỏi một câu về số liệu trong báo cáo tài chính. Hệ thống chạy **4 tác tử** rồi trả về
**phương trình** và **đáp án**, hiện rõ từng bước.

## Trước tiên: nhập khoá API

Space này **không kèm khoá của ai cả**. Bấm **Nhập khoá API** ở đầu trang rồi điền:

- **Base URL** — endpoint tương thích OpenAI của bạn. Phải là `https`, thường kết thúc bằng `/v1`.
- **API key** — khoá của bạn.
- **Tên model** — chỉ cần nếu endpoint của bạn đặt tên khác ba model có sẵn.

Khoá **chỉ nằm trong trình duyệt của bạn** (`localStorage`), được gửi kèm mỗi lượt hỏi để
máy chủ gọi hộ, rồi bị vứt. Không ghi log, không lưu, không ai khác thấy. Bấm *Xoá khoá* để gỡ.

Không muốn phải tin lời trên thì clone về chạy trên máy mình — xem mục cuối.

## Dùng thế nào

Đầu vào có 3 phần, đúng như một dòng trong tập ViNumQA:

| Phần | Nhập ở đâu |
|---|---|
| **Ngữ cảnh** — đoạn văn quanh bảng | nút `+` → *Ngữ cảnh* |
| **Bảng** — số liệu | nút `+` → *Bảng dữ liệu* (dán Markdown / Excel / CSV) |
| **Câu hỏi** | gõ vào khung chat |

Lười nhập thì bấm một thẻ ví dụ — lấy thẳng từ `datasets/ViNumQA/origin/train.json`.

## Nó làm gì

```
Câu hỏi + tài liệu
   │
   ├─ 1. Tách câu hỏi con        chỉ tra số, không tính
   ├─ 2. Trả lời câu hỏi con      mỗi câu một lời gọi, chạy song song
   ├─ 3. Lập kế hoạch             lấy mẫu n kế hoạch tính toán khác nhau
   └─ 4. Bỏ phiếu & thực thi      kế hoạch nhiều phiếu nhất thì chạy
   │
   └─▶  subtract(7.50, 6.75)  →  0.75
```

Hai ý chính: **tách việc tìm số ra khỏi việc làm toán**, và **không tin một đường suy
luận duy nhất** — lấy nhiều kế hoạch rồi cho chúng bỏ phiếu.

Đây là cài đặt của Nguyen, Ha, Le & Vu, *"A Graph-Based Agent Approach to Numerical
Reasoning Question Answering"* ([VLSP 2025](https://aclanthology.org/2025.vlsp-1.29/)) —
hệ thắng Subtask 2 với EA 84.00%. Giữ nguyên tham số của paper: `n=15`, `T=0.6`,
`top_p=0.95`, `top_k=20`, prompt tiếng Việt, bỏ phiếu canonical.

> Hạ `n` cho chạy nhanh là đang bớt phần đa đường — thành cấu hình
> *"Decomposition only"* trong bảng ablation, không còn là hệ đầy đủ.

## Model

Ba tên có sẵn trên header: `DeepSeek-V4-Flash`, `gemma-4-31B-it`, `gpt-oss-120b` —
đây là tên trên FPT Cloud Marketplace. Endpoint khác thì điền tên model vào ô
*Tên model* trong cửa sổ khoá, nó sẽ hiện thêm trong danh sách.

Hai model reasoning trả lời chắc hơn nhưng chậm hơn nhiều. Đo thật cùng một câu:

| Model | Thời gian | Kết quả |
|---|---|---|
| `gemma-4-31B-it` | 2,7 s | `subtract(7.50, 6.75)` → `0.75` |
| `gpt-oss-120b` | 5,9 s | `subtract(7.50, 6.75)` → `0.75` |
| `DeepSeek-V4-Flash` | 28,5 s | `subtract(7.50, 6.75)` → `0.75` |

## Tự chạy

```bash
git clone https://huggingface.co/spaces/Hieu18012005/num-reasoning-vi-financial
cd num-reasoning-vi-financial
pip install -r requirements.txt
cp .env.example .env        # điền API_KEY và BASE_URL của bạn
python server.py
```

Chạy kiểu này thì khoá nằm trong `.env` trên máy bạn, giao diện không hỏi khoá nữa.
