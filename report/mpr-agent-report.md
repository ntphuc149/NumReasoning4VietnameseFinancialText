# MPR-Agent — báo cáo triển khai

Áp dụng paper **"A Graph-Based Agent Approach to Numerical Reasoning Question
Answering"** (Nguyen, Ha, Le, Vu — VLSP 2025,
[aclanthology.org/2025.vlsp-1.29](https://aclanthology.org/2025.vlsp-1.29/)) vào
repo này. Paper nhất Subtask 2 với **EA 84.00%**, và **không train gì cả** —
chỉ tái cấu trúc cách hỏi model thành một đồ thị 4 agent.

Ngày: 17/8/2026 · Model: `gemma-4-31B-it` qua `BASE_URL` (FPT AI Factory)

---

## 1. Kiến trúc

Paper gọi là "graph-based" vì có **hai** đồ thị lồng nhau; cả hai đều được
implement tường minh.

**Đồ thị 1 — pipeline DAG, 4 node:**

```
q, C ──▶ [1] SubqueryGenerator   G_sq(q, C)          SQ = {sq_1..sq_k}, k∈[3,5]
                    │ fan-out, mỗi subquery 1 call độc lập
         [2] SubqueryAnswerer    A_sq(sq_j, C)       V  = {v_1..v_k}
                    │ fan-in
         [3] Planner             P_n-sample(V,C,q,T) n = 15 kế hoạch ứng viên
                    │
         [4] EquationExtractor   canonicalize → vote → p* → Execute → a*
```

**Đồ thị 2 — plan DAG:** prompt yêu cầu *"maximum parallelization"*, nên bản
thân mỗi kế hoạch là một DAG tính toán; các action độc lập nằm cùng level
(`PlanGraph.levels()`).

**Tư tưởng cốt lõi:** (1) tách *tìm số* khỏi *tính toán* — 3 dòng "DO NOT" trong
prompt B.1 cưỡng chế ranh giới này; (2) không cam kết một đường suy luận —
sample 15 kế hoạch; (3) consensus thay cho critic — không cần model chấm điểm,
không cần nhãn.

---

## 2. Các file

Toàn bộ nằm trong `notebooks/vinumqa/graph-agent/`, đúng quy ước repo
(`notebooks/evaluate/scorer.py` cũng tổ chức như vậy).

```
notebooks/vinumqa/graph-agent/
├── agentic/                          8 module, mỗi module một nhiệm vụ
├── tests/test_agentic.py             53 test, KHÔNG tốn API call
├── mpr-agent-gemma-4-31b-it.ipynb    driver: smoke → full → re-vote → ablation → A/B
├── README.md                         thiết kế đầy đủ + các cảnh báo khi đọc số
└── outputs/                          trace, CSV, summary (gitignored)
```

| Module | Nhiệm vụ | Ứng với paper |
|---|---|---|
| `config.py` | `AgentConfig` / `RunConfig`, mọi tham số, ghi rõ cái nào của paper cái nào của ta | §5.1 |
| `scoring.py` | Cầu nối **duy nhất** sang `notebooks/evaluate/scorer.py` — nạp bằng importlib, không copy | — |
| `prompts.py` | Appendix B verbatim (VI + EN) + bản vá opt-in + prompt fallback | Appendix B |
| `llm.py` | Format context `C` + client OpenAI-compat, RateLimiter, probe `n>1` | — |
| `program.py` | plan DSL → `PlanGraph` → program ViNumQA → vote | §4.3, §4.4 |
| `agents.py` | `AgentState` + 4 node + `AgentGraph` | §4.1–4.4 |
| `runner.py` | Batch, checkpoint/resume, chấm điểm, `oracle@n`, re-vote offline | — |

**Không sửa file cũ nào** ngoài `.gitignore` (thêm `graph-agent/outputs`).

---

## 3. Chỗ khó nhất: plan DSL ≠ program ViNumQA

Planner nói ngôn ngữ của paper; `scorer.py` chấm ngôn ngữ khác. Paper **không mô
tả** cách chuyển đổi của họ, nên `program.py` PART 2 là phần tái dựng — và là
nơi mọi lỗi âm thầm sẽ nằm.

| | Plan DSL (paper) | Program ViNumQA |
|---|---|---|
| Cú pháp | `1. subtract(a='21', b='47')` | `subtract(21, 47)` |
| Tham chiếu | `$1` — action id, đếm từ 1 | `#0` — vị trí step, đếm từ 0 |
| Toán tử bảng | `table_max(row_identifier='X')` — **1 arg** | `table_max(X, none)` — **2 arg** |
| Kết thúc | `join()` + `<END_OF_PLAN>` | không có |
| Toán tử | 8 + join | 10 (thêm `exp`, `greater`) |

Ba luật đáng chú ý nhất trong 7 luật:

- **`table_*(row, none)`** — verify trên dữ liệu: cả **63/63** lời gọi `table_*`
  trong gold test đều có arg thứ hai đúng là `none`. Nhãn hàng emit **không bọc
  nháy**, vì nhãn thật có ngoặc (`ROE (%)`, `P/E (x)`, `EPS (VND)`) — tokenizer
  bracket-aware của `scorer.py` xử lý được, nhưng nháy thì không.
- **Literal giữ nguyên verbatim.** PA của `scorer.py` chỉ nhận literal có trong
  gold, nên mọi chuẩn hoá số "cho đẹp" đều là mất PA. Gold có 18 chỗ dùng
  literal `100` hợp lệ (chỉ số gốc 100), nên không được special-case số 100.
- **Cắt nhánh chết.** Action mà đáp án không phụ thuộc vào sẽ bị prune. Không
  ảnh hưởng EA (chỉ lấy step cuối) nhưng **ảnh hưởng PA**: `equal_program` duyệt
  *mọi* step của prediction và loại nếu gặp literal không có trong gold.

`exp`/`greater`: đếm trên dữ liệu thật — `exp` **0 lần** ở cả train lẫn test,
`greater` **1 lần** ở test, **0 lần** ở train. Bộ 8 tool của paper phủ 496/497
mẫu ⇒ giữ nguyên, hai cái kia để sau flag `use_prompt_ext`.

---

## 4. Đã chạy những gì

### Tầng 1 — verify không tốn API (xong, xanh hết)

| Kiểm tra | Kết quả |
|---|---|
| `pytest notebooks/vinumqa/graph-agent/tests -q` | **53/53 PASS** |
| 4 ví dụ Figure B.11/B.12 | transpile **khớp chính xác** cả 4 |
| **Round-trip toàn bộ gold** — `transpile(parse_plan(to_plan(g))) ≡ g` | **0 lỗi / 4.069 program** (train 2993 + valid 584 + test 497) |

Round-trip là bài test mạnh nhất: ép transpiler qua hàng nghìn ca thật — đánh số
lại `#`, arity `table_*`, nhãn hàng có ngoặc, chuỗi nhiều bước — mà không tốn
một API call nào. 5 gold trong train mà chính `scorer.py` không parse được là
nhiễu dữ liệu có sẵn, được skip chứ không tính là lỗi.

### Tầng 2 — smoke run 30 mẫu (xong)

```
PA 0.6667   EA 0.7667
fallback_rate 0.033   empty_rate 0.033   mean_usable_candidates 13.9
oracle_pa 0.6667      oracle_ea 0.7667
4.75 s/mẫu   ~9.075 token/mẫu   ~4.5 request/mẫu
```

Sức khoẻ ứng viên (đo trên trace, tỉ lệ trong tổng số plan sinh ra):

| Giai đoạn | Tỉ lệ |
|---|---|
| ok (thành program chạy được) | **97.42%** |
| hỏng ở parse | 1.78% |
| hỏng ở transpile | 0.64% |
| execute ra `n/a` | 0.08% |
| gọi hàng không có trong bảng | 0.08% |

### Tầng 3 — full run `test.json` 497 mẫu

**Đang chạy** (~352/497 lúc viết). Cùng tập, cùng `scorer.py` với mọi dòng trong
bảng kết quả README.

### Tầng 4 — ablation Table 4 + A/B prompt

Đã cấu hình sẵn trong notebook, **chưa chạy**: `n_samples=1` (Decomposition
only), `use_decomposition=False` (Multi-path only), `use_prompt_ext=True` (A/B
prompt). Baseline direct-prompt đã có sẵn trong repo (dòng 0-shot/few-shot),
không cần chạy lại.

---

## 5. Về việc so điểm với paper — KHÔNG so trực tiếp được

Ba lý do, đều mang tính cấu trúc:

1. **Khác tập đánh giá.** Paper báo cáo trên public/private test của VLSP
   (Subtask 2: EA 84.00 / PA 74.07). Ta chấm trên `test.json` (497 mẫu).
   `private_test.json` trong repo có 1625 mẫu nhưng `qa` **chỉ có `question`** —
   không nhãn, không chấm được.
2. **Khác model và khác luật.** Paper dùng Qwen3-8B / Qwen3-32B chạy vLLM local.
   Luật Subtask 2 **loại trừ model API-only độc quyền** — đúng thứ ta đang dùng.
3. **Khác định nghĩa PA.** Paper §5.3: `norm(p_pred) ≡ norm(p_gold)` với "≡ là
   **string-level identity check**". `scorer.py`: **tương đương ký hiệu bằng
   sympy** cộng ràng buộc chỉ dùng literal có trong gold. Hai metric không cái
   nào bao cái nào — `scorer.py` lỏng hơn ở biến đổi đại số, chặt hơn ở literal
   mới. Con số PA hai bên **không cùng thang**.

⇒ So sánh có nghĩa duy nhất là với **baseline của chính repo**: cùng
`test.json`, cùng `scorer.py`, **cùng model**. Đó đúng là cấu trúc Table 2 của
paper (agent vs direct prompt trên cùng model).

```
gemma-4-31B-it, few-shot(3)   PA 0.5674  EA 0.6137     ← baseline hiện tại
gemma-4-31B-it, MPR-Agent     PA ?       EA ?          ← full run đang chạy
        (smoke 30 mẫu:        PA 0.667   EA 0.767)
```

---

## 6. Đánh giá độ trung thành với paper

### Đúng

4 node và vai trò từng node; toàn bộ prompt Appendix B verbatim (kể cả lỗi chính
tả của paper); `n=15`, `T=0.6`, `top_p=0.95`, `top_k=20`; planner nhận
`(V, C, q, T)` đúng eq(3); tie-break ít step hơn; inference-only; hai lớp đồ thị
tường minh.

Một trường **không** verbatim và đã ghi rõ trong `prompts.py`:
`planner_query_block`. Paper in B.7/B.9/B.11 là phần chỉ dẫn tĩnh nhưng không hề
in khối mang câu hỏi, context và câu trả lời subquery. Tôi tái dựng từ bullet
B.9 (nhắc tới mục "BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON") và format
`Truy vấn:` / `Bối cảnh:` / `Kế hoạch:` trong ví dụ B.11.

### CHƯA đúng — mức đếm phiếu

§4.4 (chữ) nói: canonicalize **program** rồi gom cụm. Đó là cái đang chạy.

Nhưng **Figure 1 ghi "Top result voting"** và vẽ mỗi ứng viên kèm *giá trị đã
thực thi*:

| Ứng viên trong Figure 1 | Giá trị |
|---|---|
| `add(19038.80, 9445.09), add(#0, 6286.89)` | 34770.78 |
| `add(19038.80, 6286.89), add(#0, 9445.09)` | 34770.78 |
| `add(19038.80, 31434.47), add(#0, 6286.89)` | 56760.16 |

Hai ứng viên đầu là **program khác nhau về cấu trúc nhưng cùng giá trị**, và
hình cho 34770.78 thắng **2–1**. Với vote theo cấu trúc, chúng rơi vào hai cụm
khác nhau ⇒ đếm ra **1–1–1**, và đa số mà hình vẽ mô tả **không bao giờ hình
thành**. Ví dụ của chính paper không tái lập được bằng chữ của chính paper.

⇒ Cơ chế paper *vẽ* là vote trên **kết quả thực thi**. Bằng chứng gián tiếp:
EA (84.00) của họ cao hơn PA (74.07) 10 điểm, và §5.5 viết *"our agent
prioritizes functional correctness over stylistic conformity"*.

**Trạng thái:** đã có test
`test_figure_1_candidates_are_not_merged_by_structure_voting` ghim hành vi hiện
tại để khi thêm `vote_mode="result"` thì đó là thay đổi có chủ ý, không phải
regression. **Chưa implement.**

Điểm làm dịu: trên `gemma-4-31B-it` lỗi này **hiện chưa ảnh hưởng số nào**, vì
mỗi mẫu chỉ sinh ra ~1 program duy nhất nên mọi kiểu vote đều không có gì để
chọn (xác nhận bằng re-vote offline: `canonical` và `symbolic` cho **kết quả
giống hệt nhau**). Nó sẽ ảnh hưởng trên model có diversity thật.

### Đính chính so với bản thiết kế ban đầu

Tôi từng viết `vote_mode="symbolic"` sẽ gộp được case Example 5.1. **Sai.**
`equal_program` chỉ nhận literal có trong program đem so, nên nó chỉ gộp *sắp
xếp lại đại số trên cùng tập literal*, không gộp được cặp Example 5.1 vì cặp đó
đưa vào literal mới. Đã sửa trong code, test và tài liệu.

---

## 7. Hai phát hiện thực nghiệm

### 7.1. n-sampling không đóng góp gì trên model này

26/30 mẫu smoke chỉ sinh ra **1 plan duy nhất** trên cả 15 lần sample, và
`oracle@15` bằng **đúng** PA. Đã loại trừ khả năng do code hoặc endpoint —
kiểm tra 3 đường trên cùng một planner prompt thật:

| Đường gọi | Số plan khác nhau |
|---|---|
| client của package này, server-side `n=15` | 1 / 15 |
| gọi thẳng OpenAI SDK, server-side `n=15` | 1 / 15 |
| 15 request riêng biệt, song song | 1 / 15 |
| *(đối chứng)* prompt mở, `n=15`, cùng tham số | **9 / 15** |

Nâng riêng nhiệt độ planner (8 mẫu):

| T planner | plan khác nhau | **program khác nhau** | PA | EA | oracle PA |
|---|---|---|---|---|---|
| 0.6 (paper) | 1.2 | 0.9 | 0.625 | 0.750 | 0.625 |
| 0.9 | 1.8 | 0.9 | 0.625 | 0.750 | 0.625 |
| 1.2 | 1.5 | 0.9 | 0.625 | 0.750 | 0.625 |

Plan thô có biến động nhưng sau transpile **quy về cùng một program**. Sau khi
decomposition đã chốt các con số, kế hoạch gần như bị xác định.

Đối chiếu: paper báo n-sampling là contribution chính, +5.6 EA / +9.3 PA ở 8B.
Chênh lệch có thể do Qwen3 đa dạng hơn gemma ở cùng nhiệt độ. Chưa kiểm chứng.

### 7.2. Trung thành prompt phải trả giá bằng PA

B.9/B.10 bảo: gặp `48.8%` thì bỏ dấu % và dùng `48.8`. Nhưng gold ViNumQA viết
phần trăm *dùng như tỷ lệ* thành thập phân: `divide(333, 0.159)`,
`divide(9151, 0.741)` — **34/497** gold test có literal `0.xx` kiểu này.

Theo đúng prompt, planner sinh `divide(15.9, 100), divide(333, #0)`: **EA đúng,
PA sai** (literal `15.9` và `100` không có trong gold). Đây chính xác là
Example 5.1 của paper, và nhiều khả năng là lý do PA (74.07) của họ thấp hơn
EA (84.00) tới 10 điểm. Mặc định giữ prompt verbatim; `prompts.py` có bản vá
1 dòng sau flag `use_prompt_ext` để A/B đo đúng cái giá này.

---

## 8. Quyết định ngoài paper (đều có flag, để đo được chứ không trộn vào kết quả)

| Flag | Mặc định | Lý do |
|---|---|---|
| `drop_invalid_candidates` | `True` | Vote trên program không chạy được thì một cụm hỏng có thể thắng cụm chạy được |
| `validate_row_labels` | `True` | `table_*` gọi hàng không có trong bảng luôn ra `n/a`; bắt sớm để thành lỗi có tên thay vì điểm 0 |
| `use_direct_prompt_fallback` | `True` | Prediction rỗng chắc chắn 0 điểm cả hai metric. Mọi lần dùng đều ghi lại → `fallback_rate` báo cùng PA/EA |
| `keep_all_candidates` | `True` | Cần cho `oracle@n` **và** cho re-vote offline |
| `use_prompt_ext` | `False` | Giữ run chính trung thành prompt |
| `temperature_planner` | `None` | `None` = dùng nhiệt độ chung 0.6 của paper |

**`oracle@n` — chỉ số paper cần nhưng không báo.** Tách được hai chế độ hỏng ở
§5.5.2 của họ: `oracle_pa − PA` = phần voting **vứt đi** (heuristic selection
error); `1 − oracle_pa` = phần model **chưa bao giờ sinh ra** (systematic
reasoning error).

**Re-vote offline.** Vì mọi ứng viên được lưu kèm `program` và `exe_result`, đổi
cách chọn người thắng **không tốn API call nào** (`runner.revote()`). Đây cũng
là cách sẽ đánh giá `vote_mode="result"` khi implement, trên chính run đã có.

---

## 9. Chi phí

Endpoint hỗ trợ `n>1` phía server (đã probe: 15 choice trong 1 request) ⇒
**~4,5 request và ~9k token/mẫu**. Nếu không hỗ trợ thì ~20 request/mẫu;
`llm.py` tự probe một lần lúc khởi động rồi cache.

Full 497 mẫu: ~4,5M token, ~2.200 request, ~40–55 phút ở 4 worker. Checkpoint
mỗi mẫu, resume được.

---

## 10. Việc còn lại

1. Kết thúc full run 497 → PA/EA + `oracle@15`, so với `gemma-4-31B-it`
   few-shot(3) (**PA 0.5674 / EA 0.6137**).
2. **Implement `vote_mode="result"`** (mục 6) → đánh giá bằng re-vote offline,
   không tốn API.
3. Chạy ablation Table 4 (2 run) → định lượng đóng góp thật của decomposition và
   multi-path trên model này.
4. Chạy A/B `use_prompt_ext=True` → đo giá của việc trung thành prompt.
5. Cập nhật README gốc + `progress-log.md` sau khi có số cuối (chưa làm, chờ ý
   anh).
