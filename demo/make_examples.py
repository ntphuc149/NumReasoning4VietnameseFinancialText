"""Rút ví dụ cho demo trực tiếp từ tập train của ViNumQA.

    python demo/make_examples.py                 # 120 mẫu -> demo/examples.json
    python demo/make_examples.py --n 300 --split valid

Không sửa nội dung mẫu: `pre_text`, `post_text`, `table`, `question` giữ nguyên
văn, chỉ lọc theo kích thước bảng để thẻ ví dụ trên giao diện không bị tràn, và
kèm `program`/`exe_ans` để về sau muốn đối chiếu đáp án thì đã có sẵn.

`server.py` đọc `examples.json` trước; không có thì đọc thẳng
`datasets/ViNumQA/origin/train.json`. File này chỉ để bản demo tách rời khỏi
repo vẫn còn ví dụ thật mà không phải mang theo tập train 76 MB.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Rút ví dụ từ tập train ViNumQA")
    ap.add_argument("--n", type=int, default=120, help="số mẫu giữ lại")
    ap.add_argument("--split", default="train", help="train | valid | test")
    ap.add_argument("--seed", type=int, default=20250824)
    args = ap.parse_args()

    src = ROOT / "datasets" / "ViNumQA" / "origin" / f"{args.split}.json"
    if not src.exists():
        raise SystemExit(f"Không thấy {src}")

    data = json.loads(src.read_text(encoding="utf-8"))

    def fits(s: dict) -> bool:
        table = s.get("table") or []
        if not (3 <= len(table) <= 9):
            return False
        if not table or not (2 <= len(table[0]) <= 5):
            return False
        question = (s.get("qa") or {}).get("question", "")
        return 10 <= len(question) <= 130

    pool = [s for s in data if fits(s)]
    random.Random(args.seed).shuffle(pool)

    items = [{
        "id": s.get("id", ""),
        "pre": "\n".join(s.get("pre_text") or []),
        "post": "\n".join(s.get("post_text") or []),
        "table": s["table"],
        "query": s["qa"]["question"],
        "program": s["qa"].get("program", ""),
        "exe_ans": s["qa"].get("exe_ans", ""),
    } for s in pool[: args.n]]

    out = HERE / "examples.json"
    out.write_text(
        json.dumps({"source": f"datasets/ViNumQA/origin/{args.split}.json",
                    "count": len(items), "items": items},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"examples.json  {len(items)} mau  {size:,.0f} KB  (tu {len(data)} dong {args.split})")


if __name__ == "__main__":
    main()
