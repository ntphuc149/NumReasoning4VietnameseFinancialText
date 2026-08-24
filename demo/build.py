"""Gộp demo thành một file tự chứa để publish/chia sẻ.

    python demo/build.py

Đọc index.html + styles.css + data.js + app.js, nội tuyến toàn bộ và ghi ra
preview.html. Bản gộp không có <!doctype>/<html>/<head>/<body> vì host Artifact
tự bọc phần khung đó; mở trực tiếp bằng trình duyệt vẫn chạy bình thường.
"""
import io, re
from pathlib import Path

HERE = Path(__file__).parent

def read(name):
    return io.open(HERE / name, encoding="utf-8").read()

html = read("index.html")
title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
fonts = "\n".join(l for l in html.splitlines() if l.strip().startswith("<link rel=\"pre")
                  or "fonts.googleapis" in l)
body = html[html.index("<body>") + 6 : html.index("<script src=\"data.js\">")].strip()

out = "\n".join([
    f"<title>{title}</title>",
    fonts,
    "",
    "<style>",
    read("styles.css").strip(),
    "</style>",
    "",
    body,
    "",
    "<script>",
    read("data.js").strip(),
    "",
    read("app.js").strip(),
    "</script>",
    "",
])
io.open(HERE / "preview.html", "w", encoding="utf-8").write(out)
print("preview.html  {:,} bytes".format(len(out.encode("utf-8"))))
