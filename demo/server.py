"""Backend cho demo — chạy pipeline MPR-Agent thật và stream từng chặng.

    python demo/server.py            # http://localhost:8000
    python demo/server.py --port 9000 --model DeepSeek-V4-Flash

Không viết lại pipeline. Nạp thẳng `agentic` từ notebooks/vinumqa/graph-agent/
và chạy `build_default_graph()` — đúng 4 node, đúng prompt B.1-B.12, đúng cách
bỏ phiếu §4.4. Khác biệt duy nhất so với `Runner`: chạy từng node một để bắn
sự kiện SSE ra giao diện sau mỗi node, thay vì gọi `graph.run()` một phát.

API_KEY và BASE_URL đọc từ .env ở gốc repo. Khoá không bao giờ ra tới trình duyệt.
Chỉ dùng thư viện chuẩn + openai (đã có sẵn trong requirements.txt).
"""

from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
import os
import random
import queue
import re
import socket
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AGENT_DIR = ROOT / "notebooks" / "vinumqa" / "graph-agent"

# ------------------------------------------------------------------ env --
def load_env() -> None:
    """`demo/.env` trước, rồi `.env` ở gốc repo. File đầu tiên tìm thấy thắng."""
    for path in (HERE / ".env", ROOT / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def bind_scorer() -> None:
    """Trỏ `agentic.scoring` vào bản `scorer.py` nằm cạnh file này.

    `agentic/scoring.py` nạp scorer theo đường dẫn, dò ngược lên thư mục có
    `.git` — tách demo/ ra khỏi repo là gãy. Nó trả về sớm nếu `vinumqa_scorer`
    đã có trong `sys.modules`, nên đăng ký trước là đủ; `agentic/` giữ nguyên
    văn, không phải sửa một dòng nào.
    """
    local = HERE / "scorer.py"
    if not local.exists() or "vinumqa_scorer" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("vinumqa_scorer", local)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["vinumqa_scorer"] = module
    spec.loader.exec_module(module)


load_env()
bind_scorer()
# Bản `agentic/` kèm theo demo được ưu tiên; không có thì lấy bản gốc trong repo.
sys.path.insert(0, str(HERE if (HERE / "agentic").is_dir() else AGENT_DIR))

from agentic.agents import build_default_graph          # noqa: E402
from agentic.config import AgentConfig                  # noqa: E402
from agentic.llm import LLMClient, hidden_reasoning     # noqa: E402
from agentic.runner import build_state                  # noqa: E402
from agentic.scoring import execute_program             # noqa: E402

MODEL = os.environ.get("DEMO_MODEL", "DeepSeek-V4-Flash")

# Ba model có trên endpoint và đủ khoẻ cho pipeline này. gemma-4-31B-it là
# baseline in-context mạnh nhất của repo, nên để cạnh hai model reasoning thì
# so được đóng góp của kiến trúc với đóng góp của model — đúng so sánh Bảng 2.
MODELS = ["DeepSeek-V4-Flash", "gemma-4-31B-it", "gpt-oss-120b"]

# Chạy công khai (Hugging Face Space chẳng hạn) thì mỗi lượt hỏi là tiền trong
# tài khoản API của người dựng. Đặt DEMO_RATE_LIMIT để chặn bớt; 0 = không giới
# hạn, hợp cho lúc chạy trên máy mình.
RATE_LIMIT = int(os.environ.get("DEMO_RATE_LIMIT", "0"))   # lượt/giờ mỗi IP
MAX_N = int(os.environ.get("DEMO_MAX_N", "15"))            # trần cho n_samples
_hits: dict[str, list[float]] = {}
_hits_lock = threading.Lock()

# DEMO_BYOK=1: máy chủ không mang khoá nào, mỗi người xem tự nhập khoá của họ.
# Khoá đi thẳng từ trình duyệt vào lời gọi rồi bị vứt — không ghi log, không lưu.
BYOK = os.environ.get("DEMO_BYOK", "").lower() in ("1", "true", "yes")

# Chỉ dùng khi muốn ghim endpoint; để trống thì nhận mọi tên miền công khai.
ALLOWED_HOSTS = [h.strip().lower()
                 for h in os.environ.get("DEMO_ALLOWED_HOSTS", "").split(",") if h.strip()]


class DemoError(RuntimeError):
    """Lỗi có câu chữ dành cho người dùng — hiện nguyên văn, không kèm tên lớp."""


def check_base_url(url: str) -> str:
    """Chặn SSRF. Trả về chuỗi rỗng nếu hợp lệ, ngược lại là lý do từ chối.

    Nhận `base_url` từ trình duyệt biến máy chủ thành proxy cho người lạ, nên
    phải chặn https-only và không cho trỏ vào mạng nội bộ.
    """
    u = urllib.parse.urlparse(url or "")
    if u.scheme != "https":
        return "BASE_URL phải bắt đầu bằng https://"
    host = (u.hostname or "").lower()
    if not host:
        return "BASE_URL thiếu tên miền"
    if ALLOWED_HOSTS and host not in ALLOWED_HOSTS:
        return "Chỉ chấp nhận endpoint thuộc: " + ", ".join(ALLOWED_HOSTS)
    try:
        infos = socket.getaddrinfo(host, 443)
    except socket.gaierror:
        return f"Không phân giải được tên miền {host}"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast):
            return "BASE_URL trỏ vào địa chỉ nội bộ"
    return ""


def rate_ok(ip: str) -> tuple[bool, int]:
    """Cửa sổ trượt 1 giờ. Trả về (cho phép, số giây phải chờ)."""
    if RATE_LIMIT <= 0:
        return True, 0
    now = time.time()
    with _hits_lock:
        seen = [t for t in _hits.get(ip, []) if now - t < 3600]
        if len(seen) >= RATE_LIMIT:
            _hits[ip] = seen
            return False, int(3600 - (now - seen[0])) + 1
        seen.append(now)
        _hits[ip] = seen
        return True, 0


# ------------------------------------------------------------- ví dụ --
# examples.json (rút từ train bằng make_examples.py) trước cho nhanh; không có
# thì đọc thẳng tập train. Cả hai đều là dòng nguyên văn của ViNumQA.
EXAMPLE_PATHS = [
    HERE / "examples.json",
    ROOT / "datasets" / "ViNumQA" / "origin" / "train.json",
]
_examples: tuple[list, str] | None = None


def load_examples() -> tuple[list, str]:
    global _examples
    if _examples is not None:
        return _examples
    for path in EXAMPLE_PATHS:
        if not path.exists():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):                      # examples.json
            _examples = (raw.get("items", []), raw.get("source", path.name))
        else:                                          # train.json nguyên bản
            _examples = ([{
                "id": s.get("id", ""),
                "pre": "\n".join(s.get("pre_text") or []),
                "post": "\n".join(s.get("post_text") or []),
                "table": s.get("table") or [],
                "query": (s.get("qa") or {}).get("question", ""),
                "program": (s.get("qa") or {}).get("program", ""),
                "exe_ans": (s.get("qa") or {}).get("exe_ans", ""),
            } for s in raw], str(path.relative_to(ROOT)).replace("\\", "/"))
        return _examples
    _examples = ([], "")
    return _examples


def pick_examples(k: int = 3) -> dict:
    items, source = load_examples()
    usable = [x for x in items
              if x.get("table") and 3 <= len(x["table"]) <= 9
              and 2 <= len(x["table"][0]) <= 5
              and 10 <= len(x.get("query", "")) <= 130]
    picks = random.sample(usable, min(k, len(usable))) if usable else []
    return {"source": source, "total": len(usable), "items": picks}


# ------------------------------------------------------- pipeline chạy --
def make_config(model: str, n_samples: int) -> AgentConfig:
    """Mặc định của paper, chỉ đổi model và n."""
    return AgentConfig(
        model_subquery_gen=model,
        model_subquery_ans=model,
        model_planner=model,
        model_fallback=model,
        n_samples=n_samples,          # paper: 15
        temperature=0.6,              # paper: 0.6
        top_p=0.95,                   # paper: 0.95
        top_k=20,                     # paper: 20
        prompt_lang="vi",             # phụ lục B.1/B.3/B.5... là tiếng Việt
        vote_mode="canonical",        # §4.4
        # Bản demo bỏ hai node phân rã (§4.1/§4.2) — đúng dòng "Multi-path only"
        # trong bảng ablation của paper. Generator return ngay, answerer thấy
        # rỗng cũng return, planner bỏ hẳn khối câu hỏi con: không tốn lời gọi
        # nào. Theo số của paper, bỏ phần này chỉ mất ~0,1-0,4 EA.
        use_decomposition=False,
    )


def to_sample(payload: dict) -> dict:
    """Ba phần trên giao diện -> đúng shape một dòng ViNumQA."""
    def lines(text: str) -> list[str]:
        return [p.strip() for p in (text or "").split("\n") if p.strip()]

    table = payload.get("table") or []
    return {
        "id": "demo",
        "pre_text": lines(payload.get("pre", "")),
        "post_text": lines(payload.get("post", "")),
        "table": [[str(c) for c in row] for row in table],
        "qa": {"question": payload.get("query", "").strip()},
    }


NUM_RE = re.compile(r"-?\d[\d.,]*\d|-?\d")


def to_float(token: str):
    """`100,00`, `100.00`, `1.234,56`, `1,234.56` -> float. None nếu không phải số."""
    t = token.strip().rstrip(".,")
    if not t:
        return None
    has_dot, has_comma = "." in t, "," in t
    if has_dot and has_comma:                       # cái đứng sau là dấu thập phân
        dec = "." if t.rfind(".") > t.rfind(",") else ","
        t = t.replace("," if dec == "." else ".", "").replace(dec, ".")
    elif has_comma:                                 # 1,234 = nghìn / 100,00 = thập phân
        tail = t.rsplit(",", 1)[1]
        t = t.replace(",", "") if len(tail) == 3 else t.replace(",", ".")
    elif has_dot:
        tail = t.rsplit(".", 1)[1]
        if len(tail) == 3 and t.count(".") >= 1 and len(t.split(".")[0]) <= 3 and t.count(".") > 1:
            t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def cells_matching(wanted, table) -> list[list[int]]:
    """Toạ độ các ô có giá trị số trùng với một trong `wanted`."""
    hits: list[list[int]] = []
    for r, row in enumerate(table):
        if r == 0:
            continue
        for c, cell in enumerate(row):
            if c == 0:
                continue
            cell_vals = [to_float(t) for t in NUM_RE.findall(str(cell))]
            cell_vals = [v for v in cell_vals if v is not None]
            if any(abs(cv - w) < 1e-9 for cv in cell_vals for w in wanted):
                hits.append([r, c])
    return hits[:8]


def split_steps(program: str, table_raw) -> list[dict]:
    """Chạy từng tiền tố của chương trình để lấy giá trị trung gian #0, #1…"""
    parts = [p.strip() for p in re.split(r",\s*(?=[a-z_]+\()", program) if p.strip()]
    steps = []
    for i in range(len(parts)):
        prefix = ", ".join(parts[: i + 1])
        ok, value = execute_program(prefix, table_raw)
        steps.append({
            "ref": f"#{i}",
            "expr": parts[i],
            "out": str(value) if ok else "n/a",
        })
    return steps


# Node nào được hiện thành một chặng trên giao diện. Hai node phân rã đã tắt
# nên không có mặt ở đây — chúng vẫn nằm trong đồ thị nhưng no-op.
# (tên hiện trên giao diện, phụ đề). Phụ đề để rỗng thì giao diện không dựng
# dòng đó — điền lại chuỗi vào đây là nó hiện lên như cũ.
STAGES = {
    "planner": ("Lập kế hoạch", ""),
    "equation_extractor": ("Bỏ phiếu & thực thi", ""),
}

REF_RE = re.compile(r"#\d+")


def program_hits(program: str, table) -> list[list[int]]:
    """Ô nào trong bảng mang đúng con số mà chương trình thắng cuộc đã dùng.

    Trước đây suy ra từ câu trả lời của các câu hỏi con; bỏ phân rã rồi thì lấy
    thẳng từ literal trong chương trình — chính xác hơn, vì đó đúng là những số
    đi vào phép tính.
    """
    body = REF_RE.sub(" ", program or "")
    wanted = [v for v in (to_float(t) for t in NUM_RE.findall(body)) if v is not None]
    return cells_matching(wanted, table)


def capture_reasoning(client: LLMClient) -> list[str]:
    """Giữ lại chuỗi suy nghĩ mà pipeline vốn vứt đi.

    `agentic/llm.py::_content()` chỉ lấy `content`; phần suy nghĩ nằm ở
    `reasoning_content` (DeepSeek, GLM) hoặc `reasoning` (gpt-oss) và bị bỏ —
    đúng như nó nên làm, vì kế hoạch mới là thứ pipeline cần.

    Bọc ngay lời gọi SDK của riêng client này thay vì sửa file vendored: nhờ
    vậy `agentic/` giữ nguyên văn, và mỗi lượt chạy có sổ riêng nên hai người
    hỏi cùng lúc không lẫn vào nhau. Model không phải loại reasoning thì danh
    sách trả về rỗng.
    """
    traces: list[str] = []
    inner = client.client.chat.completions.create

    def create(*args, **kwargs):
        response = inner(*args, **kwargs)
        # Bỏ qua lệnh thăm dò server-side-n ("Say OK.", max_tokens=8): nó cũng
        # sinh reasoning nhưng chẳng liên quan gì tới câu hỏi của người dùng.
        if int(kwargs.get("max_tokens") or 0) > 16:
            for choice in getattr(response, "choices", None) or []:
                text = hidden_reasoning(getattr(choice, "message", None))
                if text:
                    traces.append(text.strip())
        return response

    client.client.chat.completions.create = create
    return traces


def candidate_rows(state) -> list[dict]:
    """Từng đường suy luận: kế hoạch model viết ra, chương trình, kết quả."""
    rows = []
    for c in state.candidates:
        rows.append({
            "i": c.index,
            "plan": (c.raw_plan or "").strip(),
            "program": c.program or "",
            "ok": bool(c.usable),
            "result": "" if c.exe_result is None else str(c.exe_result),
            "error": c.error or "",
        })
    return rows


def run_pipeline(payload: dict, emit) -> None:
    n_samples = int(payload.get("n_samples") or 15)
    n_samples = max(1, min(n_samples, MAX_N))

    # Khoá của người xem (chế độ BYOK) thắng khoá của máy chủ, nếu có gửi lên.
    api_key = (payload.get("api_key") or "").strip() or os.environ.get("API_KEY", "")
    base_url = (payload.get("base_url") or "").strip() or os.environ.get("BASE_URL", "")
    if not api_key or not base_url:
        raise DemoError("Chưa có khoá API. Bấm “Khoá API” trên đầu trang để nhập.")
    if payload.get("base_url"):
        why = check_base_url(base_url)
        if why:
            raise DemoError(why)

    # Khoá của người xem thì endpoint cũng của họ — tên model không ép theo
    # danh sách của mình được. Chỉ chặn chuỗi rác.
    model = (payload.get("model") or "").strip()
    if payload.get("api_key"):
        if not model or len(model) > 80 or not re.fullmatch(r"[\w.:/-]+", model):
            model = MODEL
    elif model not in MODELS:
        model = MODEL

    config = make_config(model, n_samples)
    client = LLMClient(config, api_key=api_key, base_url=base_url)
    reasoning = capture_reasoning(client)
    sample = to_sample(payload)
    state = build_state(sample, lang=config.prompt_lang)

    graph = build_default_graph(client, config)
    order = graph.order()
    visible = [n.name for n in order if n.name in STAGES]
    index_of = {name: i for i, name in enumerate(visible)}

    emit({"type": "run_start", "model": model, "n_samples": n_samples,
          "vote_mode": config.vote_mode, "prompt_lang": config.prompt_lang,
          "decomposition": config.use_decomposition,
          "stages": [{"key": k, "name": STAGES[k][0], "sub": STAGES[k][1]}
                     for k in visible]})

    for node in order:
        key = node.name
        i = index_of.get(key)

        # Node đã tắt thì chạy im lặng — nó no-op, không gọi API, không hiện lên.
        if i is None:
            state = node.run(state)
            continue

        emit({"type": "stage", "i": i, "key": key, "status": "start"})
        state = node.run(state)
        seconds = round(state.traces[-1].seconds, 2)

        if key == "planner":
            emit({"type": "plans", "sampled": len(state.raw_plans)})
            if reasoning:
                # Ghép được với từng kế hoạch chỉ khi endpoint trả n lựa chọn
                # trong một lần gọi — lúc đó thứ tự khớp. Nếu phải bắn n lệnh
                # song song thì thứ tự về là thứ tự xong, không ghép bừa.
                paired = (client.supports_server_side_n(model)
                          and len(reasoning) == len(state.raw_plans))
                emit({"type": "reasoning", "items": reasoning[:n_samples],
                      "paired": paired})
            emit({"type": "stage", "i": i, "key": key, "status": "done",
                  "seconds": seconds,
                  "note": f"n = {n_samples}, T = {config.temperature}"})

        elif key == "equation_extractor":
            vote = state.vote
            clusters = sorted((c.count for c in vote.clusters), reverse=True) if vote else []
            program = state.program or ""
            usable = state.usable_candidates
            emit({"type": "hits", "cells": program_hits(program, sample["table"])})
            emit({"type": "candidates", "items": candidate_rows(state),
                  "winner": program})
            emit({
                "type": "final",
                "program": program,
                "answer": "" if state.answer is None else str(state.answer),
                "steps": split_steps(program, state.table_raw) if program else [],
                "votes": clusters or [len(usable)],
                "distinct": len(clusters),
                "plans": len(state.raw_plans),
                "usable": len(usable),
                "consensus": round(vote.consensus, 3) if vote else 0.0,
                "fallback": state.fallback,
                "seconds": seconds,
            })
            emit({"type": "stage", "i": i, "key": key, "status": "done",
                  "seconds": seconds,
                  "note": "đồng thuận" if not state.fallback else "dự phòng"})

    usage = client.usage.as_dict() if hasattr(client, "usage") else {}
    emit({"type": "run_end", "errors": list(state.errors), "usage": usage})


# ----------------------------------------------------------------- HTTP --
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            sys.stderr.write("  %s %s\n" % (self.command, self.path))

    # --- GET /api/health -------------------------------------------------
    def do_GET(self):
        route = self.path.split("?")[0]

        if route == "/api/health":
            has_server_key = bool(os.environ.get("API_KEY") and os.environ.get("BASE_URL"))
            return self.json_out({
                "ok": has_server_key or BYOK,
                "byok": BYOK and not has_server_key,
                "model": MODEL,
                "models": MODELS,
                "max_n": MAX_N,
                # Không lộ endpoint của máy chủ ra ngoài khi chạy công khai.
                "endpoint": "" if BYOK else os.environ.get("BASE_URL", ""),
            })

        if route == "/api/examples":
            query = urllib.parse.urlparse(self.path).query
            k = urllib.parse.parse_qs(query).get("k", ["3"])[0]
            try:
                k = max(1, min(int(k), 12))
            except ValueError:
                k = 3
            return self.json_out(pick_examples(k))

        super().do_GET()

    def fail(self, code: int, message: str) -> None:
        """Báo lỗi bằng JSON.

        `send_error()` nhét message vào dòng trạng thái HTTP, mà dòng đó chỉ
        mã hoá được latin-1 — tiếng Việt có dấu làm nó ném UnicodeEncodeError
        và rớt luôn kết nối. Giữ dòng trạng thái ASCII, đưa chữ xuống body.
        """
        body = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        # File tĩnh không cache: sửa CSS/JS xong chỉ cần F5, khỏi Ctrl+F5.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def json_out(self, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- POST /api/run  (SSE) --------------------------------------------
    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.fail(400, "JSON không hợp lệ")
            return
        if not payload.get("query", "").strip():
            self.fail(400, "Thiếu câu hỏi")
            return

        # Sau proxy của Hugging Face thì IP thật nằm ở X-Forwarded-For.
        fwd = self.headers.get("X-Forwarded-For", "")
        ip = fwd.split(",")[0].strip() or self.client_address[0]
        allowed, retry = rate_ok(ip)
        if not allowed:
            self.fail(429, f"Quá số lượt cho phép. Thử lại sau {retry // 60 + 1} phút.")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        events: queue.Queue = queue.Queue()

        def emit(obj):
            events.put(obj)

        def work():
            try:
                run_pipeline(payload, emit)
            except DemoError as exc:
                events.put({"type": "error", "message": str(exc)})
            except Exception as exc:                       # noqa: BLE001
                traceback.print_exc()
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(None)

        threading.Thread(target=work, daemon=True).start()

        while True:
            item = events.get()
            if item is None:
                break
            try:
                self.wfile.write(
                    b"data: " + json.dumps(item, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionAbortedError):
                return


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser(description="Demo server cho MPR-Agent")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="0.0.0.0 khi chạy trong container")
    ap.add_argument("--model", default=MODEL, choices=MODELS)
    args = ap.parse_args()
    MODEL = args.model

    missing = [k for k in ("API_KEY", "BASE_URL") if not os.environ.get(k)]
    if missing and not BYOK:
        sys.exit(f"Thiếu {', '.join(missing)} — đặt trong .env, biến môi trường, "
                 f"hoặc bật DEMO_BYOK=1 để người xem tự nhập khoá")

    items, source = load_examples()
    print(f"  model     {MODEL}  (đổi được trên giao diện: {', '.join(MODELS)})")
    print(f"  ví dụ     {len(items)} mẫu từ {source or 'không có — dùng bản kèm trong data.js'}")
    if BYOK and missing:
        print("  khoá      BYOK — mỗi người xem tự nhập khoá của họ")
    else:
        print(f"  endpoint  {os.environ['BASE_URL']}")
    if RATE_LIMIT:
        print(f"  giới hạn  {RATE_LIMIT} lượt/giờ mỗi IP, n tối đa {MAX_N}")
    print(f"  demo      http://{args.host}:{args.port}\n")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
