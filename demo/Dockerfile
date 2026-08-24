# Hugging Face Space (SDK: docker). Space lắng nghe cổng 7860.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF chạy container bằng user 1000, không phải root.
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# BYOK mac dinh: Space khong mang khoa nao, moi nguoi xem tu nhap khoa cua ho.
# Muon Space dung khoa cua ban: dat secret API_KEY + BASE_URL trong Settings,
# them variable DEMO_BYOK=0, va nho dat DEMO_RATE_LIMIT.
ENV HOST=0.0.0.0 \
    DEMO_BYOK=1 \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 7860
CMD ["python", "server.py"]
