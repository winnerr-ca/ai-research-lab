# CPU image for rlcore training runs.
#
# Build:  docker build -t rlcore:cpu .
# Run:    docker run --rm -v "$(pwd)/outputs:/app/outputs" rlcore:cpu \
#             algo=ppo algo.total_updates=5
#
# Outputs land in the mounted ./outputs directory; checkpoints inside a
# run directory survive container restarts the same way (mount, then
# resume with resume_from=/app/outputs/<...>/checkpoint.pt).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-install-project && uv pip install --no-deps -e .

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["rlcore-train"]
