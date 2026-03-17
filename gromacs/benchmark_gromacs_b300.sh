#!/bin/bash
# ============================================================
# GROMACS 2024 ApoA1 Benchmark — runs INSIDE the container
# Called by run_gromacs_b300.sh via docker exec
# ============================================================
set -uo pipefail

GMX="${GMX:-/usr/local/gromacs/bin/gmx}"
APOA1_DIR="/benchmarks/apoa1"
OUT_DIR="${OUT_DIR:-/workspace/logs/gromacs}"
GPU_ID="${GPU_ID:-0}"
OMP_THREADS="${OMP_THREADS:-16}"
NSTEPS="${NSTEPS:-50000}"
RESETSTEP="${RESETSTEP:-10000}"

sep() { printf '=%.0s' {1..60}; echo; }

mkdir -p "${OUT_DIR}"

sep
echo "  GROMACS 2024 — ApoA1 Benchmark (92,224 atoms)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  GROMACS: $(${GMX} --version 2>&1 | grep 'GROMACS version' | head -1)"
sep
echo ""

# ── Step 1: grompp — generate .tpr run input ─────────────────────────────────
echo ">>> grompp (generating run input)..."
${GMX} grompp \
    -f "${APOA1_DIR}/pme.mdp" \
    -c "${APOA1_DIR}/apoa1.gro" \
    -p "${APOA1_DIR}/apoa1.top" \
    -o "${OUT_DIR}/apoa1.tpr" \
    -maxwarn 5 \
    2>&1 | tail -5

# ── Step 2: mdrun — production run ───────────────────────────────────────────
echo ""
echo ">>> mdrun (${NSTEPS} steps, GPU offload: nb+pme+bonded)..."
${GMX} mdrun \
    -s "${OUT_DIR}/apoa1.tpr" \
    -ntmpi 1 \
    -ntomp "${OMP_THREADS}" \
    -nb gpu \
    -pme gpu \
    -bonded gpu \
    -gpu_id "${GPU_ID}" \
    -nsteps "${NSTEPS}" \
    -resetstep "${RESETSTEP}" \
    -noconfout \
    -dlb no \
    -g "${OUT_DIR}/gromacs_b300_run.log" \
    -deffnm "${OUT_DIR}/gromacs_b300" \
    2>&1 | grep -E "Performance|step|hours"

# ── Parse result ──────────────────────────────────────────────────────────────
echo ""
sep
echo "  RESULTS"
sep

PERF_LINE=$(grep "^Performance:" "${OUT_DIR}/gromacs_b300_run.log" | tail -1)
NS_DAY=$(echo "${PERF_LINE}" | awk '{print $2}')
HR_NS=$(echo "${PERF_LINE}" | awk '{print $3}')

printf "  %-20s %s ns/day  (%s hours/ns)\n" "B300 SXM6 (measured):" "${NS_DAY}" "${HR_NS}"
echo ""
echo "  Reference (single GPU, GROMACS 2024, ApoA1):"
printf "  %-20s %s\n" "H100 SXM5:" "~450 ns/day"
printf "  %-20s %s\n" "B200 SXM:"  "~585 ns/day (estimated)"
printf "  %-20s %s\n" "B300 SXM6:" "${NS_DAY} ns/day  ← this run"
sep
echo ""

# Save JSON
python3 -c "
import json, sys
data = {'ns_day': '${NS_DAY}', 'hours_ns': '${HR_NS}',
        'nsteps': ${NSTEPS}, 'resetstep': ${RESETSTEP},
        'system': 'ApoA1 (92224 atoms)', 'gpu': '$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)'}
with open('${OUT_DIR}/gromacs_b300_result.json', 'w') as f:
    json.dump(data, f, indent=2)
print('Result saved to ${OUT_DIR}/gromacs_b300_result.json')
"
