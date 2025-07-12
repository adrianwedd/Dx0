FROM python:3.11-slim AS builder
WORKDIR /build

# Install development dependencies for running tests and building a wheel
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source and build a wheel
COPY . .
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir dist \
    && pip install --no-cache-dir dist/*.whl

# Run the test suite using the built package
RUN pytest -q

FROM python:3.11-slim AS runtime
WORKDIR /app

# Install only runtime dependencies from the built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp

CMD ["sdb-cli", "--help"]
