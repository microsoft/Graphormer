FROM python:3.9.19-slim-bookworm@sha256:69e712dbe4c4a166527cbf69374533125cfb6ee93a5e39031a0191c741d386d7 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    DGLBACKEND=pytorch \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/graphormer \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/graphormer
COPY . .

RUN test -f fairseq/setup.py \
    || (echo "Initialize submodules before building: git submodule update --init --recursive" >&2; exit 1)

RUN python -m pip install \
        pip==23.3.2 \
        setuptools==69.5.1 \
        wheel==0.48.0 \
    && python -m pip install \
        torch==1.9.1+cu111 \
        torchaudio==0.9.1 \
        --find-links https://download.pytorch.org/whl/cu111/torch_stable.html \
    && python -m pip install --requirement docker/requirements.txt \
    && python -m pip install \
        dgl==0.7.2 \
        --find-links https://data.dgl.ai/wheels/repo.html

COPY docker/sitecustomize.py /usr/local/lib/python3.9/site-packages/sitecustomize.py

RUN cd fairseq \
    && CFLAGS="-include cstdint" CXXFLAGS="-include cstdint" \
        python -m pip install --no-build-isolation .

RUN python - <<'PY'
import numpy
import pyximport

pyximport.install(
    setup_args={"include_dirs": numpy.get_include()},
    inplace=True,
    language_level=3,
)
import graphormer.data.algos
PY

RUN python docker/verify_environment.py


FROM python:3.9.19-slim-bookworm@sha256:69e712dbe4c4a166527cbf69374533125cfb6ee93a5e39031a0191c741d386d7 AS runtime

LABEL org.opencontainers.image.title="Graphormer legacy runtime" \
      org.opencontainers.image.description="Pinned Python 3.9, PyTorch 1.9, and Fairseq environment for Graphormer" \
      org.opencontainers.image.source="https://github.com/microsoft/Graphormer" \
      org.opencontainers.image.version="legacy-py39-torch1.9"

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DGLBACKEND=pytorch \
    PYTHONPATH=/opt/graphormer \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/graphormer /opt/graphormer

WORKDIR /opt/graphormer

CMD ["bash"]
