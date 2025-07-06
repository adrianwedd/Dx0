FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
RUN pip install --no-cache-dir -e .
RUN pytest -q

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app
CMD ["python", "cli.py", "--help"]
