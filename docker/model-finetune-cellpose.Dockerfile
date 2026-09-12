# BioEngine worker image with the model-finetune app's Cellpose backend
# preinstalled, and only that backend enabled (MODEL_FINETUNE_BACKENDS=cellpose).
# Ray's runtime_env venv inherits the image's site-packages, so at deploy time
# every cellpose pin resolves as already satisfied and the app skips the
# multi-GB download/install that otherwise runs on first startup; and with the
# micro-sam backend switched off it fits a single GPU (see apps/model-finetune
# README, issues #0060/#0061/#0062).
#
# Mirrors docker/model-runner.Dockerfile layer-for-layer: install the app
# requirements FIRST (largest, least-changing layer), then requirements-worker,
# then the bioengine package --no-deps, then Ray last via RAY_VERSION. Keep in
# sync with worker.Dockerfile when the worker build changes.
#
# numpy: the cellpose runtime hard-pins numpy==1.26.4 (cellpose is irreconcilable
# with micro-sam, which needs numpy>=2 — that is why the two backends are
# separate Ray deployments). requirements-worker.txt installs AFTER and also pins
# numpy==1.26.4, so there is no conflict and 1.26.4 survives; the build asserts
# this (scripts/build_model_finetune_cellpose.sh). Only the cellpose backend is
# baked, so requirements-runtime.txt (micro-sam) is intentionally NOT installed.
#
# Versioned by the MODEL-FINETUNE APP version (apps/model-finetune/manifest.yaml),
# not the BioEngine version — the app's pins are what this image exists to
# preinstall. Each tag is built against one BioEngine version, recorded in the
# io.bioengine.version label and BIOENGINE_VERSION env var.
#
# Build (from the repo root) via scripts/build_model_finetune_cellpose.sh, which
# fills both version args from the checkout. By hand:
#   docker build \
#       -f docker/model-finetune-cellpose.Dockerfile \
#       --build-arg MODEL_FINETUNE_VERSION=<app-version> \
#       --build-arg BIOENGINE_VERSION=<bioengine-version> \
#       -t model-finetune-cellpose:<app-version> .
#
# Rebuild whenever apps/model-finetune/requirements-{entry,runtime-cellpose}.txt
# change, or whenever the BioEngine code the app runs on changes (the bioengine/
# package, requirements-worker.txt, or the Ray pin). A BioEngine-only change
# still needs a new model-finetune app version to be publishable at all; the
# build script refuses to overwrite a published tag.

# Rolling tag — each build picks up current Debian-slim security patches.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Model-finetune cellpose-backend dependencies — installed before everything
# else so this multi-GB layer survives worker-requirement and bioengine updates.
# requirements-worker.txt is installed after; both pin numpy==1.26.4, so there
# is no conflict. The micro-sam runtime (requirements-runtime.txt) is not baked.
COPY apps/model-finetune/requirements-entry.txt \
     apps/model-finetune/requirements-runtime-cellpose.txt \
     /app/model-finetune/
RUN pip install -U pip && \
    pip install -r model-finetune/requirements-entry.txt \
                -r model-finetune/requirements-runtime-cellpose.txt

# Worker requirements — intentionally does NOT pin Ray (installed last, below).
COPY requirements-worker.txt /app/
RUN pip install -r requirements-worker.txt

COPY bioengine/ /app/bioengine/
COPY pyproject.toml README.md LICENSE /app/

# Install the bioengine package without dependencies — all runtime deps are
# already satisfied by requirements-worker.txt.
RUN pip install --no-deps .

# Ray install — kept as the final step so RAY_VERSION can be overridden at build
# time without invalidating any prior layer cache.
ARG RAY_VERSION=2.55.1
RUN pip install "ray[client,serve]==${RAY_VERSION}" "protobuf>=4,<7"

ENV BIOENGINE_RAY_VERSION=${RAY_VERSION}

# Only the cellpose backend runs from this image. The baked ENV is the value
# seen by the Ray node worker on a single-container worker; startup deployments
# should ALSO pass application_env_vars={"*": {"MODEL_FINETUNE_BACKENDS":
# "cellpose"}} as the node-agnostic vector.
ENV MODEL_FINETUNE_BACKENDS=cellpose

# Version metadata last, so bumping either version rebuilds nothing but this
# layer. org.opencontainers.image.source links the GHCR package to this repo.
ARG MODEL_FINETUNE_VERSION=unknown
ARG BIOENGINE_VERSION=unknown
ENV BIOENGINE_MODEL_FINETUNE_VERSION=${MODEL_FINETUNE_VERSION} \
    BIOENGINE_VERSION=${BIOENGINE_VERSION}
LABEL org.opencontainers.image.source=https://github.com/aicell-lab/bioengine \
      org.opencontainers.image.version=${MODEL_FINETUNE_VERSION} \
      io.bioengine.version=${BIOENGINE_VERSION}

CMD [ "/bin/bash" ]
