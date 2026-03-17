#!/bin/bash
# ============================================================
# AlphaFold2 Inference Benchmark — NVIDIA B300 SXM6
#
# Runs JAX-based AlphaFold2 Evoformer proxy benchmark (no weights
# required) and reports time-to-structure vs H100/B200.
#
# Usage:
#   bash run_alphafold_b300.sh                         # GPU 4 (default)
#   CUDA_GPU=0 bash run_alphafold_b300.sh              # GPU 0
#   AF_WEIGHTS_DIR=/path/to/weights bash run_...       # AF2 weights
#
# AlphaFold2 weights (free download, ~3.5 GB):
#   https://github.com/google-deepmind/alphafold#accessing-the-alphafold-model-parameters
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="b300-alphafold2:latest"
CUDA_GPU="${CUDA_GPU:-4}"
AF_WEIGHTS_DIR="${AF_WEIGHTS_DIR:-${SCRIPT_DIR}/weights}"
LOG_DIR="${SCRIPT_DIR}/logs"

# ── Build image if missing ────────────────────────────────────────────────────
if ! sg docker -c "docker image inspect ${IMAGE}" &>/dev/null 2>&1; then
    echo "==> Building ${IMAGE} from Dockerfile.alphafold ..."
    sg docker -c "docker build -f ${SCRIPT_DIR}/Dockerfile.alphafold \
        -t ${IMAGE} ${SCRIPT_DIR}"
fi

mkdir -p "${LOG_DIR}"

echo ""
echo "================================================================"
echo "  AlphaFold2 Inference — NVIDIA B300 SXM6 (JAX NGC 25.01)"
echo "  $(date)"
echo "================================================================"
echo ""

sg docker -c "docker run --rm \
    --gpus '\"device=${CUDA_GPU}\"' \
    --ipc=host \
    --ulimit memlock=-1 \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda' \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v ${SCRIPT_DIR}:/workspace \
    -v ${AF_WEIGHTS_DIR}:/data/alphafold_weights:ro \
    -w /workspace \
    ${IMAGE} python3 /workspace/benchmark_alphafold_b300.py" \
    2>&1 | tee "${LOG_DIR}/alphafold_b300_run.log"

echo ""
echo "================================================================"
echo "  Log: ${LOG_DIR}/alphafold_b300_run.log"
echo "  To run full AF2 prediction (requires weights):"
echo "    AF_WEIGHTS_DIR=/path/to/weights CUDA_GPU=${CUDA_GPU} \\"
echo "    bash ${SCRIPT_DIR}/run_alphafold_b300.sh"
echo "================================================================"
