# syntax=docker/dockerfile:1
# CPU (default):
#   docker build -t pokezero .
# CUDA-compatible (same code, CUDA PyTorch wheels on an NVIDIA base):
#   docker build --build-arg BASE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 \
#                --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 -t pokezero:cuda .
#   docker run --gpus all -v $PWD/runs:/app/runs pokezero:cuda train --config configs/default.yaml
# The training loop currently runs on CPU; the CUDA image exists so batched GPU inference can be
# added later without changing the base. UNTESTED in this repo's CI (no Docker daemon was available
# when it was written) -- see README "Known gaps".
ARG BASE=python:3.12-slim
FROM ${BASE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv/bin/python --index-strategy unsafe-best-match --extra-index-url ${TORCH_INDEX} -e .
ENV PATH="/opt/venv/bin:${PATH}"

# Artifacts (checkpoints, replay data, metrics) go to a mounted volume; no secrets are baked in.
VOLUME ["/app/runs"]
ENTRYPOINT ["pokezero"]
CMD ["train", "--config", "configs/smoke.yaml"]
