#!/bin/bash
#
# Build (and optionally publish) the cellpose-only model-finetune worker image
# by hand.
#
# This is the BioEngine worker image with the model-finetune app's Cellpose
# backend preinstalled and only that backend enabled
# (docker/model-finetune-cellpose.Dockerfile), so a worker launched from it
# runs model-finetune as a single-GPU, cellpose-only startup app without any
# deploy-time package installation. It is deliberately NOT built by
# docker-publish-worker.yml — build it on demand, like the model-runner image.
#
# Rebuild when either half of what is baked in changes:
#   * the app's cellpose pins — apps/model-finetune/requirements-entry.txt and
#     requirements-runtime-cellpose.txt
#   * the BioEngine code the app runs on — the bioengine/ package,
#     requirements-worker.txt, or the Ray pin
# A BioEngine-only change still needs a new model-finetune app version: the tag
# is the app version, so there is no other way to publish it. The push guard
# below enforces that.
#
# Published as a SEPARATE GHCR package so it never pollutes the worker image's
# tag list or scan findings. The tag is the MODEL-FINETUNE APP version read from
# apps/model-finetune/manifest.yaml. The BioEngine version each tag is built
# against is read from pyproject.toml and baked in as the io.bioengine.version
# label.
#
# Usage:
#   scripts/build_model_finetune_cellpose.sh [--push]
#
# Environment overrides:
#   IMAGE        image name (default ghcr.io/aicell-lab/model-finetune-cellpose)
#   TAG          image tag  (default: version from apps/model-finetune/manifest.yaml)
#   RAY_VERSION  Ray to bake (default: the Dockerfile's pinned version)
#   FORCE        set to 1 to overwrite an already-published tag
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_FINETUNE_VERSION="$(grep -E '^version\s*:' "$PROJECT_ROOT/apps/model-finetune/manifest.yaml" \
    | sed -E 's/version\s*:\s*"?([^"]*)"?/\1/' | head -1)"
BIOENGINE_VERSION="$(grep -E '^version\s*=' "$PROJECT_ROOT/pyproject.toml" \
    | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1)"

IMAGE="${IMAGE:-ghcr.io/aicell-lab/model-finetune-cellpose}"
TAG="${TAG:-$MODEL_FINETUNE_VERSION}"
REF="${IMAGE}:${TAG}"

BUILD_ARGS=(
    --build-arg "MODEL_FINETUNE_VERSION=${MODEL_FINETUNE_VERSION}"
    --build-arg "BIOENGINE_VERSION=${BIOENGINE_VERSION}"
)
if [[ -n "${RAY_VERSION:-}" ]]; then
    BUILD_ARGS+=(--build-arg "RAY_VERSION=${RAY_VERSION}")
fi

PUSH=""
[[ "${1:-}" == "--push" ]] && PUSH=1

# Refuse to overwrite a published tag. A tag is one immutable (app pins,
# BioEngine code) pair — silently replacing it means a cluster pinned to that
# tag gets different code on its next pull, with nothing in the version to show
# for it.
if [[ -n "$PUSH" && "${FORCE:-}" != "1" ]]; then
    if docker manifest inspect "$REF" >/dev/null 2>&1; then
        cat >&2 <<EOF
${REF} is already published.

Bump 'version' in apps/model-finetune/manifest.yaml and re-run. This applies
even when only BioEngine changed: the tag is the app version, so a new
BioEngine build has no other way to be published.

FORCE=1 overwrites the tag — only for a build known to be byte-identical.
EOF
        exit 1
    fi
fi

echo "Building ${REF} from ${PROJECT_ROOT} (BioEngine ${BIOENGINE_VERSION})"
docker build \
    -f "$PROJECT_ROOT/docker/model-finetune-cellpose.Dockerfile" \
    -t "$REF" \
    "${BUILD_ARGS[@]}" \
    "$PROJECT_ROOT"

echo "Built ${REF}"

# The cellpose pin (numpy==1.26.4) must survive the worker-requirements layer,
# and the full stack must import under it — the whole point of the image is that
# the deploy-time install short-circuits, so a broken pin would only surface at
# first fine-tune. Assert both before declaring the build good.
echo "Validating numpy pin and stack imports in ${REF}"
docker run --rm "$REF" python -c "
import numpy; assert numpy.__version__ == '1.26.4', numpy.__version__
import torch, cellpose, ray, bioengine
from importlib.metadata import version
print('numpy', numpy.__version__, '| torch', torch.__version__, '| cellpose', version('cellpose'), '| ray', ray.__version__, '| bioengine', version('bioengine'))
"
echo "Validation OK"

if [[ -n "$PUSH" ]]; then
    echo "Pushing ${REF}"
    docker push "$REF"
else
    echo "Not pushed. Re-run with --push to publish."
fi
