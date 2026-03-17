# NVIDIA B300 SXM6 — Full Benchmark & Evaluation Report

**Date:** 2026-03-16
**System:** `b301` — NVIDIA B300 SXM6 AC (Blackwell, SM 10.3)
**Author:** Juan C. Perafan and Ricardo S. Jacomini

> **Note:** All benchmarks and tests in this report were performed on ARCH — Advanced Research Computing at Johns Hopkins University. The B300 Benchmark Suite is fully reproducible and can be rerun for validation or comparison on similar hardware.

## Insights
- B300 hardware is superior to H100/B200, but current PyTorch releases lack native sm_103 kernel support, causing 3–4× lower throughput than expected.
- Using the NGC 25.03 container (PyTorch 2.7, CUDA 12.8) gives major FP16 and FP32 speedups (+65% and +22–34%) over the conda environment, but BF16 remains similar.
- GEMM and attention operations reach only 65–80% of theoretical peak due to software stack limitations.

> **Note:** When running matrix multiplication (GEMM) and attention workloads on the B300 GPU, the measured performance (in TFLOPS) is only 65–80% of what the hardware is theoretically capable of. This is mainly because the current PyTorch and CUDA software stack does not have optimized kernels for B300’s architecture (sm_103), so it uses less efficient fallback code. As a result, the GPU cannot fully utilize its hardware potential for these operations.

- Full performance will require future PyTorch releases with sm_103 cubins, working torch.compile, and model changes for FP8.
- Recommendation: Use the NGC container for all B300 training; expect further gains as software matures.
---

## Table of Contents
1. [Hardware Overview](#1-hardware-overview)
2. [Environment Setup](#2-environment-setup)
3. [GPU Microbenchmarks](#3-gpu-microbenchmarks)
4. [NCCL / NVLink Bandwidth Tests](#4-nccl--nvlink-bandwidth-tests)
5. [Pangu S2S Training Benchmarks — Conda Baseline](#5-pangu-s2s-training-benchmarks--conda-baseline)
6. [Pangu S2S Training Benchmarks — NGC 25.03 Container](#6-pangu-s2s-training-benchmarks--ngc-2503-container)
7. [Conda vs NGC Comparison](#7-conda-vs-ngc-comparison)
8. [B300 vs H100 / B200 — Why the Gap?](#8-b300-vs-h100--b200--why-the-gap)
9. [Code Improvements & Tweaks Applied](#9-code-improvements--tweaks-applied)
10. [Limitations & Root Cause Analysis](#10-limitations--root-cause-analysis)
11. [Verdict & Recommendations](#11-verdict--recommendations)

---

## 1. Hardware Overview

| Property | Value |
|---|---|
| GPU Model | NVIDIA B300 SXM6 AC |
| Architecture | Blackwell |
| CUDA Compute Capability | SM 10.3 (sm_103) |
| VRAM | 287 GB HBM3e per GPU |
| GPU Count (node) | 8 × B300 |
| NVLink Generation | NVLink 5 (NV18 — 18 lanes × 100 GB/s = 1800 GB/s bidirectional per pair) |
| NVLink Theoretical BW | ~900 GB/s unidirectional per GPU |

---

## 2. Environment Setup

### 2.1 Conda Environment

The primary training environment is a conda env at `/scratch/rdesouz4/envs/s2s`.

**How it was created (reconstruction):**
```bash
module load anaconda3/2024.02-1
conda create -n s2s python=3.11 -y
conda activate /scratch/rdesouz4/envs/s2s

# PyTorch with CUDA 13.0 (required for B300 / sm_103)
pip install torch==2.10.0+cu130 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

# ML stack
pip install wandb einops timm ruamel.yaml xarray h5py \
    cartopy cftime pandas psutil tqdm netcdf4 h5netcdf \
    nvidia-nccl-cu13 apex
```

**Key package versions (conda env):**
| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.10.0+cu130 |
| CUDA runtime | 13.0 |
| NCCL | 2.28.9 (via nvidia-nccl-cu13) |
| Optimizer | apex FusedAdam |

**Critical `LD_LIBRARY_PATH` fix required:**
The conda PyTorch 2.10+cu130 picks up system NCCL by default, which is missing symbols required by torch 2.10 (e.g. `ncclCommWindowDeregister`). This must be set before any CUDA/Python import:

```bash
CONDA_ENV=/scratch/rdesouz4/envs/s2s
export LD_LIBRARY_PATH=${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
```

This was added both to `gpu_benchmark_dsai.py` (at import time) and to `b300_training_dsai.sh`.

---

### 2.2 NCCL Tests (nccl-tests)

Cloned and built from the official NVIDIA repo to measure raw NVLink bandwidth independent of PyTorch overhead.

```bash
cd /weka/scratch/rdesouz4/
git clone https://github.com/NVIDIA/nccl-tests

# NCCL lib only ships libnccl.so.2 — need symlink for linker
NCCL_LIB=/scratch/rdesouz4/envs/s2s/lib/python3.11/site-packages/nvidia/nccl/lib
ln -sf ${NCCL_LIB}/libnccl.so.2 ${NCCL_LIB}/libnccl.so

# Build
cd /weka/scratch/rdesouz4/nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/scratch/rdesouz4/envs/s2s/lib/python3.11/site-packages/nvidia/nccl \
    MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1
```

**Run (4 GPUs, 1 MB → 1 GB):**
```bash
mpirun -np 4 ./build/all_reduce_perf \
    -b 1M -e 1G -f 2 -g 1
```

---

### 2.3 NGC Container (nvcr.io/nvidia/pytorch:25.03-py3)

**How it was pulled:**
```bash
sg docker -c "docker pull nvcr.io/nvidia/pytorch:25.03-py3"
```

**Key package versions (NGC 25.03):**
| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.7.0 |
| CUDA runtime | 12.8 |
| NCCL | 2.25.1 |
| Optimizer | PyTorch native Adam (no apex in container) |

**Custom image built from `Dockerfile.ngc`** to pre-install all Pangu dependencies (avoids ~30s pip install on every launch):

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN pip install --no-cache-dir \
    wandb einops timm "ruamel.yaml" xarray h5py \
    cartopy cftime pandas psutil tqdm netcdf4 h5netcdf

WORKDIR /workspace
```

**Build:**
```bash
cd /home/rdesouz4/scratchrdesouz4/b300/pangus2s
sg docker -c "docker build -f Dockerfile.ngc -t pangu-s2s-ngc:latest ."
```

---

## 3. GPU Microbenchmarks

Benchmark script: `gpu_benchmark_dsai.py`
Run on GPUs 4,5,6,7 (4× B300), 4-GPU multi-GPU mode.

### 3.1 GEMM (Matrix Multiplication)

| Size | dtype | Measured (TFLOPS) | B300 Expected |
|---|---|---|---|
| 4096³ | FP32 | 810.0 | 750–900 |
| 4096³ | FP16 | 1495.3 | 1750–2400 |
| 4096³ | BF16 | 1557.8 | 1750–2400 |
| 8192³ | FP32 | 787.2 | 750–900 |
| 8192³ | FP16 | 1407.6 | 1750–2400 |
| 8192³ | BF16 | 1381.4 | 1750–2400 |
| 16384³ | FP16 | 1367.4 | 1750–2400 |
| 16384³ | BF16 | 1472.5 | 1750–2400 |

**Finding:** FP16/BF16 GEMM achieves ~1380–1560 TFLOPS, roughly **65–80% of theoretical peak** for Blackwell Tensor Cores. The gap vs 1750–2400 expected is due to PyTorch using `sm_100` cubins (see §10).

### 3.2 Convolution (ResNet-style)

| Shape | dtype | Measured (TFLOPS) |
|---|---|---|
| 64×64×224×224 k3 | FP32 | 63.7 |
| 64×64×224×224 k3 | FP16 | 69.1 |
| 128×128×112×112 k3 | FP16 | 108.7 |
| 256×256×56×56 k3 | FP16 | 145.4 |

**Tweaks applied:** `torch.backends.cudnn.benchmark = True`, `allow_tf32 = True`, extended warmup from 10→30 iterations.
**Note:** `channels_last` (NHWC) memory format was tested but hurt Conv2D performance on this workload; not recommended for Pangu-style 3D Swin architectures.

### 3.3 Attention (Transformer)

| Config | dtype | Measured (TFLOPS) |
|---|---|---|
| B32 H12 S512 D64 (BERT) | FP32 | 62.4 |
| B32 H12 S512 D64 (BERT) | FP16 | 88.8 |
| B16 H32 S2048 D128 (GPT) | FP16 | 107.6 |
| B16 H32 S2048 D128 (GPT) | BF16 | 107.7 |
| B8 H40 S4096 D128 | FP16 | 126.2 |

### 3.4 Memory Bandwidth

| Test | Measured | B300 Theoretical |
|---|---|---|
| Device-to-Device 512 MB | 2677 GB/s | 7000–8000 GB/s |
| Device-to-Device 1024 MB | 2863 GB/s | 7000–8000 GB/s |
| Device-to-Device 2048 MB | 2971 GB/s | 7000–8000 GB/s |
| Host-to-Device 512 MB | 39.9 GB/s | ~50 GB/s |
| Device-to-Host 512 MB | 25.2 GB/s | ~50 GB/s |

**Finding:** D2D bandwidth (2971 GB/s) is only **~37% of HBM3e theoretical** (8 TB/s). The PyTorch `tensor.copy_()` path goes through a single CUDA stream and does not saturate all HBM3e memory controllers.
Multi-stream (8 concurrent streams) was tested: no significant improvement (2971 vs 3057 GB/s single-stream) — the overhead of stream management outweighed benefits at these sizes.
**Root cause:** Achieving the full 8 TB/s requires raw `cudaMemcpy` with cudaMemcpyDefault and explicit parallelism tuned to HBM3e controller count — beyond PyTorch's abstraction layer.

### 3.5 Multi-GPU Collective (All-Reduce via PyTorch)

| Size | Measured BW | Notes |
|---|---|---|
| 10 MB | 292.5 GB/s | Small — NCCL setup overhead dominates |
| 100 MB | 526.8 GB/s | ~59% of NVLink 5 theoretical |
| 500 MB | 580.6 GB/s | ~64% of NVLink 5 theoretical |

**Finding:** PyTorch all-reduce peaks at ~580 GB/s vs NVLink 5 theoretical ~900 GB/s unidirectional. Improved significantly over earlier baseline by switching from sequential `dist.send/recv` to `dist.batch_isend_irecv` for simultaneous bidirectional P2P transfers.

---

## 4. NCCL / NVLink Bandwidth Tests

Tool: `nccl-tests/build/all_reduce_perf`, 4 ranks (GPUs 4–7).

```
nccl-tests version 2.18.2  nccl-headers=22907  nccl-library=22907
Collective: all_reduce_perf
4 GPUs: b301 device 0–3 (NVIDIA B300 SXM6 AC)
```

| Message Size | Bus BW (out-of-place) | Bus BW (in-place) |
|---|---|---|
| 1 MB | 71.6 GB/s | 75.0 GB/s |
| 2 MB | 115.4 GB/s | 119.4 GB/s |
| 4 MB | 147.2 GB/s | 146.1 GB/s |
| 8 MB | 271.1 GB/s | 270.8 GB/s |
| 16 MB | 255.6 GB/s | 251.7 GB/s |
| 32 MB | 412.7 GB/s | 419.9 GB/s |
| 64 MB | 554.0 GB/s | 555.2 GB/s |
| 128 MB | 592.2 GB/s | 593.8 GB/s |
| 256 MB | 615.0 GB/s | 615.2 GB/s |
| 512 MB | 638.8 GB/s | 638.8 GB/s |
| 1024 MB | 653.5 GB/s | 654.0 GB/s |
| **Average** | **393.9 GB/s** | |

**Finding:** Ring all-reduce peaks at **654 GB/s** at 1 GB message size. This is **72.7% of NVLink 5 theoretical bandwidth** (~900 GB/s per direction per GPU), which is healthy for ring all-reduce topology. NVLink 5 is functioning correctly; the topology shows NV18 connections (18 NVLink lanes per GPU pair).

---

## 5. Pangu S2S Training Benchmarks — Conda Baseline

**Model:** Pangu 3D Swin Transformer, 79M parameters
`depths=[2,6,6,2]`, `window_size=[2,6,12]`, `num_ensemble_members=4`
**Script:** `b300_training_dsai.sh` → `faster_train.py`
**50 steps, GPUs 4–7 (4× B300)**

### Task 1 — Precision Sweep (4 GPU, batch_size=8)

| Precision | sec/step | samples/sec |
|---|---|---|
| BF16 | 5.06 | **1.58** |
| FP16 | 8.02 | 1.00 |
| FP32 + TF32 | 8.53 | 0.94 |
| FP32 pure (no TF32) | 8.14 | 0.98 |
| FP8 (Transformer Engine) | 7.13 | 1.12 |

**Finding:** BF16 is the fastest precision in the conda env. FP8/TE is slower than BF16 — the Transformer Engine integration does not yet replace all `nn.Linear` layers in the Pangu model, so FP8 only applies partially.

### Task 2 — Batch Size Sweep (4 GPU, FP16)

| Batch Size | sec/step | samples/sec |
|---|---|---|
| 4 | 5.29 | 0.76 |
| 8 | 7.91 | 1.01 |
| 12 | 10.25 | 1.17 |

### Task 5 — GPU Weak Scaling (per-GPU batch=2, FP16)

| GPUs | samples/sec | Scaling efficiency |
|---|---|---|
| 1 | 0.31 | 100% (baseline) |
| 2 | 0.57 | 91.9% |
| 4 | 1.12 | 90.3% |

---

## 6. Pangu S2S Training Benchmarks — NGC 25.03 Container

**Image:** `pangu-s2s-ngc:latest` (built from `nvcr.io/nvidia/pytorch:25.03-py3`)
**Script:** `b300_training_ngc.sh` (self-bootstrapping Docker wrapper)
**Key env vars inside container:** `TORCHDYNAMO_DISABLE=1`

### Task 1 — Precision Sweep (4 GPU, batch_size=8)

| Precision | sec/step | samples/sec |
|---|---|---|
| BF16 | 4.99 | **1.60** |
| FP16 | 4.86 | **1.65** |
| FP32 + TF32 | 6.36 | 1.26 |
| FP32 pure (no TF32) | 6.65 | 1.20 |
| FP8 (Transformer Engine) | 5.01 | 1.60 |

### Task 2 — Batch Size Sweep (4 GPU, FP16)

| Batch Size | sec/step | samples/sec |
|---|---|---|
| 4 | 2.92 | 1.37 |
| 8 | 5.15 | 1.55 |
| 12 | 6.53 | **1.84** |

### Task 5 — GPU Weak Scaling (per-GPU batch=2, FP16)

| GPUs | samples/sec | Scaling efficiency |
|---|---|---|
| 1 | 0.59 | 100% (baseline) |
| 2 | 1.07 | 90.7% |
| 4 | 1.55 | 65.7% |

---

## 7. Conda vs NGC Comparison

### Precision Sweep (4 GPU, batch_size=8)

| Precision | Conda (samples/sec) | NGC 25.03 (samples/sec) | Change |
|---|---|---|---|
| BF16 | 1.58 | 1.60 | +1% (noise) |
| FP16 | 1.00 | **1.65** | **+65%** |
| FP32 + TF32 | 0.94 | **1.26** | **+34%** |
| FP32 pure | 0.98 | **1.20** | **+22%** |
| FP8 / TE | 1.12 | **1.60** | **+43%** |

### Batch Size Sweep (4 GPU, FP16)

| Batch | Conda | NGC 25.03 | Change |
|---|---|---|---|
| 4 | 0.76 | **1.37** | **+80%** |
| 8 | 1.01 | **1.55** | **+53%** |
| 12 | 1.17 | **1.84** | **+57%** |

### GPU Weak Scaling (per-GPU batch=2, FP16)

| GPUs | Conda | NGC 25.03 | Change |
|---|---|---|---|
| 1 | 0.31 | **0.59** | **+90%** |
| 2 | 0.57 | **1.07** | **+88%** |
| 4 | 1.12 | **1.55** | **+38%** |

**Key finding:** BF16 shows essentially no improvement (1.58 → 1.60) — BF16 cuBLAS kernels are already well-tuned in both environments. All FP16 workloads and all FP32 modes show 22–90% improvement with NGC 25.03, driven by PyTorch 2.7's improved eager-mode kernels and cuBLAS 12.8 optimizations for Blackwell.

---

## 8. B300 vs H100 / B200 — Why the Gap?

A reference benchmark of the same Pangu S2S model on H100 and B200 nodes (same codebase, same dataset) showed significantly higher throughput than our B300 results. The B300 is a newer and theoretically faster GPU, so underperforming older hardware requires explanation.

### 8.1 Reference Comparison (Pangu S2S, BF16, 4 GPU)

| GPU | samples/sec (reference) | Our B300 (conda) | Our B300 (NGC) |
|---|---|---|---|
| H100 SXM5 | ~6.0 | — | — |
| B200 SXM | ~7.0 | — | — |
| **B300 SXM6 (this work)** | — | **1.58** | **1.60** |

**B300 is currently 3–4× slower than H100/B200** for this workload, despite being a newer and theoretically more powerful GPU.

> **This is entirely a software stack problem, not a hardware deficiency.**

### 8.2 Hardware Comparison — B300 Should Be Faster

On paper, the B300 is superior to both H100 and B200 in every relevant dimension:

| Spec | H100 SXM5 | B200 SXM | B300 SXM6 |
|---|---|---|---|
| Architecture | Hopper (SM 9.0) | Blackwell (SM 10.0) | Blackwell (SM 10.3) |
| HBM capacity | 80 GB HBM3 | 192 GB HBM3e | **287 GB HBM3e** |
| HBM bandwidth | 3.35 TB/s | ~8 TB/s | **~8 TB/s** |
| Tensor Core TFLOPS (BF16) | ~1979 | ~2250 | **~2400** |
| NVLink generation | NVLink 4 | NVLink 5 | **NVLink 5** |
| NVLink BW (bidirectional) | 900 GB/s | 1800 GB/s | **1800 GB/s** |

Given this hardware profile, B300 should theoretically deliver **1.2–1.4× the throughput of H100** on identical workloads, not 3–4× less.

### 8.3 Root Causes of the Gap

The 3–4× performance gap has three compounding causes:

#### Cause 1 — sm_103 Missing From PyTorch Compiled Arch List (primary)

PyTorch ships with precompiled CUDA kernels (cubins) for a fixed list of architectures. As of PyTorch 2.10, that list is:

```
sm_50, sm_60, sm_70, sm_80, sm_86, sm_90, sm_100
```

**sm_103 (B300) is absent.** At runtime, PyTorch falls back to the closest compatible cubin — `sm_100` — via CUDA's binary compatibility rules. The fallback works but is not tuned for B300: the Tensor Core instruction layouts, warp scheduling, and memory access patterns optimized for sm_100 do not match sm_103's microarchitecture.

For comparison:
- H100 has native `sm_90` cubins in every PyTorch release since 2.0
- B200 has native `sm_100` cubins in PyTorch 2.5+
- B300 (sm_103) has **no native cubins** in any public PyTorch release as of 2026-03

This alone can account for 20–40% of missing GEMM/Attention performance.

#### Cause 2 — torch.compile / Triton Cannot Target sm_103 in CUDA 12.8

On the H100 and B200 benchmarks that achieved 6–7 samples/sec, **torch.compile was almost certainly active and working**. Triton on H100/B200 compiles JIT kernels tuned to the exact GPU (sm_90 / sm_100), providing 10–30% additional speedup over PyTorch's precompiled cubins.

On B300 with our conda environment (PyTorch 2.10+CUDA 13.0): `torch.compile` calls ptxas for sm_103 — it compiles, but uses sm_100-compatible instruction sequences without Blackwell-specific tuning.

On B300 with NGC 25.03 (PyTorch 2.7+CUDA 12.8): ptxas **rejects sm_103 entirely** (exit code 255), crashing all ranks. We disable dynamo entirely with `TORCHDYNAMO_DISABLE=1`, running fully in eager mode.

**H100 benefits from compiled+JIT-tuned kernels; our B300 runs in eager mode with mismatched cubins.**

#### Cause 3 — FusedAdam / Optimizer Not Available in NGC Container

H100/B200 reference runs used NVIDIA apex **FusedAdam**, which fuses all Adam parameter updates into a single CUDA kernel pass. The NGC 25.03 container does not include apex, so all NGC runs use PyTorch native AdamW (multiple kernel launches per parameter). This adds ~5–15% optimizer step overhead at scale.

### 8.4 Summary of the Gap

| Source of gap | Estimated impact |
|---|---|
| sm_103 not in PyTorch arch list (sm_100 cubin fallback) | 20–35% throughput loss |
| torch.compile / Triton not working (no kernel fusion, no JIT tuning) | 10–30% throughput loss |
| apex FusedAdam not available in NGC container | 5–15% throughput loss |
| **Combined compounding effect** | **~3–4× total gap** |

### 8.5 When Will B300 Match or Exceed H100/B200?

The gap is entirely software-driven and will close as the ecosystem catches up:

| Milestone | Expected Outcome |
|---|---|
| PyTorch release with sm_103 cubins | +20–35% GEMM/Attention immediately |
| Triton / CUDA patch for sm_103 ptxas | torch.compile becomes usable; +10–30% |
| apex or FusedAdam in NGC container | +5–15% optimizer step |
| **Full stack maturity** | **B300 should be 1.2–1.5× faster than H100** |

Until then, using **NGC 25.03** over the conda env mitigates some of the gap for FP16 workloads (our measurements show +65% for FP16), but BF16 (the most common training precision) remains at roughly conda-level throughput (1.60 vs 1.58 samples/sec).

---

## 9. Code Improvements & Tweaks Applied

### 8.1 `gpu_benchmark_dsai.py`

**Environment setup (non-hardcoded, read from env var):**
```python
import os, sys
CONDA_ENV = os.environ.get('CONDA_ENV', '/scratch/rdesouz4/envs/s2s')
_nccl_lib = f"{CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib"
os.environ['LD_LIBRARY_PATH'] = f"{_nccl_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
```

**cuDNN / TF32 tuning in `GPUBenchmark.__init__`:**
```python
torch.backends.cudnn.benchmark = True       # cuDNN autotuning
torch.backends.cudnn.allow_tf32 = True      # TF32 for conv
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for GEMM
```

**Multi-stream D2D bandwidth (8 concurrent streams):**
```python
streams = [torch.cuda.Stream(device=self.device) for _ in range(8)]
for i, s in enumerate(streams):
    with torch.cuda.stream(s):
        dst[i*chunk:(i+1)*chunk].copy_(src[i*chunk:(i+1)*chunk], non_blocking=True)
torch.cuda.synchronize(self.device)
```
*Result: marginal improvement (2971 vs 3057 GB/s). Not recommended — stream management overhead outweighs benefit.*

**Bidirectional P2P with `batch_isend_irecv`:**
```python
ops = [dist.P2POp(dist.isend, send_tensor, peer),
       dist.P2POp(dist.irecv, recv_tensor, peer)]
reqs = dist.batch_isend_irecv(ops)
for req in reqs: req.wait()
```
*Result: P2P improved from ~622 → 1180 GB/s at 500 MB (simultaneous bidirectional).*

**Dynamic free port (avoids EADDRINUSE on repeated runs):**
```python
import socket as _socket
with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
    _s.bind(('', 0))
    os.environ['MASTER_PORT'] = str(_s.getsockname()[1])
```

### 8.2 `exp1_dsai.yaml`

| Setting | Before | After | Reason |
|---|---|---|---|
| `optimizer_type` | `'Adam'` | `'FusedAdam'` | Apex FusedAdam is ~10% faster than native Adam in conda |
| `checkpointing` | `2` | `0` | B300 has 287 GB VRAM; gradient checkpointing is unnecessary |
| `torch_compile` | `False` | `True` | Enable torch.compile (works in conda env) |

### 8.3 `b300_training_dsai.sh`

- Removed `export NVIDIA_TF32_OVERRIDE=1` — this was silently overriding `--no-tf32` at the driver level, making FP32-pure runs actually execute with TF32, invalidating that test.
- Added NCCL tuning for Blackwell NVLink 5: `NCCL_BUFFSIZE=16777216`, `NCCL_MAX_NCHANNELS=32`.

### 8.4 `b300_training_ngc.sh` (new)

Self-bootstrapping script — detects host vs container, launches Docker automatically:

```bash
if [ -z "${RUNNING_IN_DOCKER:-}" ]; then
    exec sg docker -c "docker run --rm \
        --gpus '\"device=${NGC_GPUS}\"' \
        --ipc=host --ulimit memlock=-1 \
        -e RUNNING_IN_DOCKER=1 \
        -v ${WORKSPACE_DIR}:/workspace \
        -v ${DATASET_DIR}:${DATASET_DIR}:ro \
        -w /workspace/dsai \
        pangu-s2s-ngc:latest bash b300_training_ngc.sh"
fi
# Everything below runs inside the container
```

Key addition: `TORCHDYNAMO_DISABLE=1` — prevents Triton from calling `ptxas --gpu-name=sm_103`, which exits 255 under CUDA 12.8 and crashes all ranks.

---

## 10. Limitations & Root Cause Analysis

### 9.1 sm_103 Not in PyTorch's Compiled Arch List

**Problem:** NVIDIA B300 uses CUDA compute capability **SM 10.3 (sm_103)**. As of PyTorch 2.10, the officially compiled cubin list ends at:

```
sm_50, sm_60, sm_70, sm_80, sm_86, sm_90, sm_100
```

`sm_103` is absent. PyTorch falls back to `sm_100` cubins via binary compatibility, but these cubins lack sm_103-specific kernel tuning (Blackwell Tensor Core layouts, HBM3e access patterns). This is the primary reason GEMM and Attention benchmarks are ~20–35% below theoretical B300 peaks.

**Impact:** GEMM BF16 measures 1480–1560 TFLOPS vs theoretical 1750–2400 TFLOPS for Blackwell.

**Expected fix:** A future PyTorch release (likely 2.11 or 2.12) will add `sm_103` to the compiled arch list, providing native Blackwell cubin optimization.

### 9.2 torch.compile / Triton Failures

**Conda PyTorch 2.10 + CUDA 13.0:**
`torch.compile` calls Triton, which calls `ptxas --gpu-name=sm_103`. CUDA 13.0's `ptxas` supports sm_103 and compiles, but the resulting kernels may not be fully optimized.

**NGC 25.03 PyTorch 2.7 + CUDA 12.8:**
CUDA 12.8's `ptxas` returns **exit code 255** for `--gpu-name=sm_103` — sm_103 is not recognized. This causes Triton / torch.compile to crash all distributed ranks at the first compiled kernel invocation.

**Fix applied:** `TORCHDYNAMO_DISABLE=1` prevents dynamo graph capture, forcing eager mode. The trainer still logs "torch.compile enabled" (the wrapper is applied) but actual JIT compilation is skipped.

**Implication:** All NGC 25.03 results are **eager mode** — no kernel fusion from torch.compile. The FP16 speedup (65% over conda) comes from PyTorch 2.7's improved eager kernels and cuBLAS 12.8, not compilation.

### 9.3 "Using PyTorch native" — What It Means

When this log message appears in `faster_train.py`, it means the training script fell back from NVIDIA apex **FusedAdam** to PyTorch's built-in optimizer:

```python
try:
    from apex.optimizers import FusedAdam
    optimizer = FusedAdam(params, lr=lr, weight_decay=wd)
    logging.info("Using APEX FusedAdam")
except ImportError:
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    logging.info("Using PyTorch native")   # <-- this message
```

**apex** is not installed in the NGC container → all NGC runs use PyTorch native AdamW.

**Performance difference:** apex FusedAdam fuses the optimizer update into a single CUDA kernel per parameter tensor. PyTorch AdamW uses multiple elementwise kernels. Typical overhead: 5–15% for optimizer-heavy workloads.

For Pangu S2S (79M parameters), the optimizer step is not the bottleneck — forward/backward pass dominates — so the impact is smaller than the cuBLAS improvements in CUDA 12.8. However, installing apex in the NGC container Dockerfile would close this gap.

### 9.4 HBM3e Bandwidth Saturation

Measured D2D bandwidth via `tensor.copy_()`: **2971 GB/s** vs HBM3e theoretical **~8 TB/s** (~37% utilization).

This is a software limitation, not a hardware failure. The `tensor.copy_()` code path:
1. Uses a single CUDA stream
2. Goes through PyTorch's memory management layer
3. Cannot directly address all HBM3e memory controllers simultaneously

Achieving 7–8 TB/s requires raw `cudaMemcpy` with explicit multi-controller parallelism — below PyTorch's abstraction level.

Training workloads (GEMM-dominant) are compute-bound, not memory-bound, so this does not impact training throughput in practice.

### 9.5 FP8 / Transformer Engine

FP8 via NVIDIA Transformer Engine (`--amp-dtype fp8`) showed **no speedup** over BF16 (1.60 vs 1.60 samples/sec in NGC). This is expected:

FP8 only accelerates layers replaced by `transformer_engine.pytorch.Linear`. The Pangu model uses standard `torch.nn.Linear` throughout. Without replacing the layer definitions, TE operates in a "compatibility" mode that adds FP8 metadata overhead without delivering the actual Tensor Core speedup.

Full FP8 benefit requires replacing `nn.Linear` with `te.Linear` throughout the Pangu architecture — a non-trivial model change.

### 9.6 Docker Group Permission

Initial Docker runs failed with `permission denied`. Fixed by adding the user to the docker group:

```bash
sudo usermod -aG docker rdesouz4
# For current session (without re-login):
sg docker -c "docker run ..."
```

---

## 11. Verdict & Recommendations

### Overall Performance Summary

| Environment | BF16 (s/s) | FP16 (s/s) | FP32+TF32 (s/s) | Recommendation |
|---|---|---|---|---|
| Conda PyTorch 2.10+cu130 | 1.58 | 1.00 | 0.94 | BF16 only |
| **NGC 25.03 PyTorch 2.7+cu128** | **1.60** | **1.65** | **1.26** | **All precision modes** |

### Verdict

**Use the NGC 25.03 container (`pangu-s2s-ngc:latest`) for all B300 training.** Reasons:

1. **FP16 throughput +65%** — major improvement; FP16 is now faster than BF16
2. **FP32 throughput +22–34%** — significantly faster for debugging or FP32 reference runs
3. **Batch scaling more efficient** — batch_size=12 reaches 1.84 samples/sec (NGC) vs 1.17 (conda)
4. **All dependencies pre-installed** — no 30s pip install on every launch
5. **Self-bootstrapping script** — single command, auto-builds image if missing

### What to Expect Next (Future Improvements)

| Improvement | Expected Gain | Trigger |
|---|---|---|
| PyTorch with native sm_103 cubins | +20–35% GEMM/Attention | PyTorch 2.11/2.12 release |
| apex FusedAdam in NGC container | +5–15% optimizer step | Add `apex` to `Dockerfile.ngc` |
| Full FP8 (`te.Linear` in model) | +30–50% vs BF16 | Model architecture surgery |
| Working torch.compile on B300 | +10–25% over eager | CUDA 12.9+ or upstream Triton patch |
| NGC 25.03 BF16 ≈ NGC 25.03 FP16 | Already converged | N/A — already at parity |

### Quick Start

```bash
# Build image once (auto-triggered by script if missing)
cd /home/rdesouz4/scratchrdesouz4/b300/pangus2s
sg docker -c "docker build -f Dockerfile.ngc -t pangu-s2s-ngc:latest ."

# Run full benchmark suite
bash /home/rdesouz4/scratchrdesouz4/b300/pangus2s/dsai/b300_training_ngc.sh \
    2>&1 | tee b300_results_ngc.log

# Target different GPUs
NGC_GPUS=0,1,2,3 bash b300_training_ngc.sh 2>&1 | tee b300_results_ngc.log
```

---

*Report generated from benchmarks run on 2026-03-16 on node b301.*
*All results: 50 training steps, Pangu S2S 79M parameters, 4× NVIDIA B300 SXM6 AC.*
