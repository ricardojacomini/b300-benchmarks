#!/bin/bash
# ============================================================
# GROMACS 2024 ApoA1 Benchmark — NVIDIA B300 SXM6
#
# Self-bootstrapping: builds Docker image if missing, then
# runs the ApoA1 MD benchmark inside the container.
#
# Usage:
#   bash run_gromacs_b300.sh              # GPU 4 (default)
#   CUDA_GPU=0 bash run_gromacs_b300.sh   # GPU 0
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="b300-gromacs:latest"
CUDA_GPU="${CUDA_GPU:-4}"
LOG_DIR="${SCRIPT_DIR}/logs/gromacs"

# ── Build image if missing ────────────────────────────────────────────────────
if ! sg docker -c "docker image inspect ${IMAGE}" &>/dev/null 2>&1; then
    echo "==> Building ${IMAGE} from Dockerfile.gromacs ..."
    sg docker -c "docker build -f ${SCRIPT_DIR}/Dockerfile.gromacs \
        -t ${IMAGE} ${SCRIPT_DIR}"
fi

mkdir -p "${LOG_DIR}"

echo ""
echo "================================================================"
echo "  GROMACS 2024 ApoA1 Benchmark — NVIDIA B300 SXM6"
echo "  $(date)"
echo "================================================================"
echo ""

sg docker -c "docker run --rm \
    --gpus '\"device=${CUDA_GPU}\"' \
    --ipc=host \
    --ulimit memlock=-1 \
    -e GPU_ID=0 \
    -e OUT_DIR=/workspace/logs/gromacs \
    -v ${SCRIPT_DIR}:/workspace \
    -w /workspace \
    ${IMAGE} bash /workspace/benchmark_gromacs_b300.sh" \
    2>&1 | tee "${LOG_DIR}/gromacs_b300_docker_run.log"

echo ""
echo "================================================================"
echo "  Log: ${LOG_DIR}/gromacs_b300_docker_run.log"
if [ -f "${LOG_DIR}/gromacs_b300_result.json" ]; then
    ns_day=$(python3 -c "import json; d=json.load(open('${LOG_DIR}/gromacs_b300_result.json')); print(d['ns_day'])")
    echo "  ApoA1 performance: ${ns_day} ns/day"
fi
echo "================================================================"
