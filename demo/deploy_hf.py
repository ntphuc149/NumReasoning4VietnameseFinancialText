"""Đẩy demo lên Hugging Face Space (SDK docker).

    huggingface-cli login          # cần token có quyền WRITE
    python demo/deploy_hf.py                      # xem trước, không đẩy
    python demo/deploy_hf.py --push               # tạo/cập nhật Space
    python demo/deploy_hf.py --push --private     # Space riêng tư

Không đẩy `.env`. Space chạy chế độ BYOK — mỗi người xem tự nhập khoá của họ,
nên Settings → Variables and secrets chỉ cần **một** biến:

    DEMO_BYOK = 1      (variable)

Tuỳ chọn thêm:

    DEMO_MAX_N          trần cho n (mặc định 15)
    DEMO_ALLOWED_HOSTS  ghim endpoint, ví dụ mkp-api.fptcloud.com
    DEMO_RATE_LIMIT     lượt/giờ mỗi IP — không cần khi BYOK vì mỗi người
                        tiêu quota của chính họ

`SPACE_README.md` được đổi tên thành `README.md` trên Space vì Hugging Face đọc
phần YAML đầu file đó để biết cấu hình.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = "Hieu18012005/num-reasoning-vi-financial"

# Đủ để Space chạy. Cố tình bỏ .env, preview.html, __pycache__.
INCLUDE_FILES = [
    "Dockerfile", "requirements.txt", ".env.example",
    "server.py", "scorer.py", "make_examples.py", "build.py",
    "index.html", "styles.css", "app.js", "data.js", "examples.json",
]
INCLUDE_DIRS = ["agentic"]
NEVER = {".env"}


def collect() -> list[tuple[Path, str]]:
    """[(đường dẫn thật, đường dẫn trên Space)] — README lấy từ SPACE_README.md."""
    out: list[tuple[Path, str]] = []

    space_readme = HERE / "SPACE_README.md"
    if not space_readme.exists():
        sys.exit("Thiếu SPACE_README.md — Hugging Face cần YAML đầu file này.")
    out.append((space_readme, "README.md"))

    for name in INCLUDE_FILES:
        path = HERE / name
        if path.exists():
            out.append((path, name))
        else:
            print(f"  bỏ qua (không có): {name}")

    for folder in INCLUDE_DIRS:
        for path in sorted((HERE / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            out.append((path, path.relative_to(HERE).as_posix()))

    for _, dest in out:
        if Path(dest).name in NEVER:
            sys.exit(f"Từ chối: {dest} không được lên Space.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Đẩy demo lên Hugging Face Space")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--push", action="store_true", help="thật sự đẩy lên")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    files = collect()
    total = sum(p.stat().st_size for p, _ in files)
    print(f"\n  Space   {args.repo}")
    print(f"  Tệp     {len(files)}  ({total / 1024:,.0f} KB)")
    for path, dest in files:
        print(f"    {dest:<34} {path.stat().st_size / 1024:>8,.1f} KB")

    if not args.push:
        print("\n  Xem trước. Thêm --push để đẩy thật.\n")
        return

    from huggingface_hub import HfApi
    api = HfApi()

    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if role != "write":
        sys.exit(f"\nToken đang là '{role}'. Cần token WRITE:\n"
                 "  https://huggingface.co/settings/tokens  ->  New token, role = Write\n"
                 "  rồi: huggingface-cli login\n")

    api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="docker",
                    private=args.private, exist_ok=True)
    print(f"\n  Space sẵn sàng: https://huggingface.co/spaces/{args.repo}")

    for path, dest in files:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                        repo_id=args.repo, repo_type="space")
        print(f"    đã đẩy  {dest}")

    print("\n  Xong. Việc còn lại — Settings > Variables and secrets:")
    print("    DEMO_BYOK = 1   (variable)")
    print("  Thiếu biến này Space sẽ không khởi động được vì không có khoá nào.\n")


if __name__ == "__main__":
    main()
