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
DEFAULT_REPO = "duonghieu18012005/num-reasoning-vi-financial"

# Đủ để Space chạy. Cố tình bỏ .env, preview.html, __pycache__.
INCLUDE_FILES = [
    "Dockerfile", "requirements.txt", ".env.example",
    "server.py", "scorer.py", "make_examples.py", "build.py",
    "index.html", "styles.css", "app.js", "data.js", "examples.json",
]
INCLUDE_DIRS = ["agentic"]
NEVER = {".env"}

# Static Space: HF chỉ phục vụ file tĩnh, không chạy Python. Giao diện tự dò
# thấy không có /api/health rồi chuyển sang chế độ xem trước; examples.json vẫn
# được đọc thẳng nên thẻ ví dụ vẫn là dòng thật của tập train.
STATIC_FILES = ["index.html", "styles.css", "app.js", "data.js", "examples.json"]


def static_readme() -> str:
    """SPACE_README.md với YAML đổi sang sdk static."""
    out = []
    for line in (HERE / "SPACE_README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("sdk:"):
            out.append("sdk: static")
        elif line.startswith("app_port:"):
            continue                      # static không có cổng
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def collect(static: bool = False) -> list[tuple[Path, str]]:
    """[(đường dẫn thật, đường dẫn trên Space)] — README lấy từ SPACE_README.md."""
    out: list[tuple[Path, str]] = []

    space_readme = HERE / "SPACE_README.md"
    if not space_readme.exists():
        sys.exit("Thiếu SPACE_README.md — Hugging Face cần YAML đầu file này.")
    out.append((space_readme, "README.md"))

    if static:
        for name in STATIC_FILES:
            path = HERE / name
            if path.exists():
                out.append((path, name))
            else:
                print(f"  bỏ qua (không có): {name}")
        return out

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


def hf_token() -> str | None:
    """HF_TOKEN từ biến môi trường, không có thì lấy trong `.env` ở gốc repo.

    Trả None để `huggingface_hub` tự dùng token đã `huggingface-cli login`.
    """
    import os
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = HERE.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Đẩy demo lên Hugging Face Space")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--push", action="store_true", help="thật sự đẩy lên")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="Static Space (miễn phí): chỉ giao diện, chế độ xem trước")
    args = ap.parse_args()

    files = collect(static=args.static)
    total = sum(p.stat().st_size for p, _ in files)
    print(f"\n  Space   {args.repo}")
    print(f"  Tệp     {len(files)}  ({total / 1024:,.0f} KB)")
    for path, dest in files:
        print(f"    {dest:<34} {path.stat().st_size / 1024:>8,.1f} KB")

    if not args.push:
        print("\n  Xem trước. Thêm --push để đẩy thật.\n")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())

    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    if role != "write":
        sys.exit(f"\nToken đang là '{role}'. Cần token WRITE:\n"
                 "  https://huggingface.co/settings/tokens  ->  New token, role = Write\n"
                 "  rồi: huggingface-cli login\n")

    sdk = "static" if args.static else "docker"

    # Repo đã có thì không gọi create_repo: tài khoản free bị chặn 402 ở bước
    # *tạo* Space docker, còn Space sẵn có vẫn đổi SDK được qua YAML trong
    # README. Bỏ qua create ở đây chính là cách chuyển static -> docker.
    try:
        api.space_info(args.repo)
        print(f"\n  Space đã có, chỉ cập nhật file (sdk -> {sdk} qua README).")
    except Exception:
        try:
            api.create_repo(repo_id=args.repo, repo_type="space", space_sdk=sdk,
                            private=args.private, exist_ok=True)
        except Exception as exc:
            # Tài khoản free bị chặn 402 ngay ở bước *tạo* Space docker, nhưng
            # Space đã tồn tại thì đổi SDK qua YAML trong README lại được. Nên
            # tạo dạng static trước rồi để README kéo nó sang docker.
            if "402" not in str(exc) or sdk != "docker":
                raise
            print("  402 khi tạo Space docker — tạo dạng static rồi đổi qua README.")
            api.create_repo(repo_id=args.repo, repo_type="space", space_sdk="static",
                            private=args.private, exist_ok=True)
    print(f"\n  Space sẵn sàng: https://huggingface.co/spaces/{args.repo}")

    for path, dest in files:
        if dest == "README.md" and args.static:
            api.upload_file(path_or_fileobj=static_readme().encode("utf-8"),
                            path_in_repo=dest, repo_id=args.repo, repo_type="space")
        else:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=dest,
                            repo_id=args.repo, repo_type="space")
        print(f"    đã đẩy  {dest}")

    if args.static:
        print("\n  Xong. Static Space không chạy Python nên giao diện ở chế độ")
        print("  xem trước: trace là mô phỏng. Ví dụ vẫn là dòng thật của tập train.\n")
    else:
        print("\n  Xong. Việc còn lại — Settings > Variables and secrets:")
        print("    DEMO_BYOK = 1   (variable)")
        print("  Dockerfile đã đặt sẵn biến này, chỉ cần khi muốn đổi khác.\n")


if __name__ == "__main__":
    main()
