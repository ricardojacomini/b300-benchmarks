# NVIDIA B300 SXM6 — Full Benchmark & Evaluation Report

**Date:** 2026-03-16
**System:** `b301` — NVIDIA B300 SXM6 AC (Blackwell, SM 10.3)
**Author:** Juan C. Perafan and Ricardo S. Jacomini

> **Note:** All benchmarks and tests in this report were performed on ARCH — Advanced Research Computing at Johns Hopkins University. The B300 Benchmark Suite is fully reproducible and can be rerun for validation or comparison on similar hardware.

## Insights
- B300 hardware is superior to H100/B200, but current PyTorch releases lack native sm_103 kernel support, causing 3–4× lower throughput than expected.
- Using the NGC 25.03 container (PyTorch 2.7, CUDA 12.8) gives major FP16 and FP32 speedups (+65% and +22–34%) over the conda environment, but BF16 remains similar.
- **PyTorch Nightly 2.12.0.dev+cu130 is now the recommended stack for B300**: delivers +25–60% over NGC 25.03 with a working torch.compile path (+18–37% additional) — even without native sm_103 cubins.
- GEMM and attention operations reach only 65–80% of theoretical peak due to software stack limitations.

> **Note:** When running matrix multiplication (GEMM) and attention workloads on the B300 GPU, the measured performance (in TFLOPS) is only 65–80% of what the hardware is theoretically capable of. This is mainly because the current PyTorch and CUDA software stack does not have optimized kernels for B300’s architecture (sm_103), so it uses less efficient fallback code. As a result, the GPU cannot fully utilize its hardware potential for these operations.

- Full peak performance will require future PyTorch releases with native sm_103 cubins and model changes for FP8/FP4.
- **MLPerf Inference** (offline scenario): 4× B300 delivers 405.6 QPS (ResNet-50 FP16) and 1524.8 QPS (BERT-Large FP16) — both VALID results.
- Recommendation: Use **PyTorch Nightly cu130** for all B300 training; NGC 25.03 as fallback if nightly instability is a concern.
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
11. [PyTorch Nightly + CUDA 13.0 Benchmark](#11-pytorch-nightly--cuda-130-benchmark)
12. [MLPerf Inference — ResNet-50 & BERT-Large Offline](#12-mlperf-inference--resnet-50--bert-large-offline)
13. [Verdict & Recommendations](#13-verdict--recommendations)

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

| GPU | samples/sec | Gap vs H100 | Notes |
|---|---|---|---|
| H100 SXM5 | ~6.0 | baseline | Native sm_90 cubins, torch.compile working |
| B200 SXM | ~7.0 | +17% over H100 | Native sm_100 cubins, torch.compile working |
| B300 SXM6 — conda 2.10+cu130 | 1.58 | **−74% vs H100** | sm_100 fallback, compile partial |
| B300 SXM6 — NGC 25.03 | 1.60 | **−73% vs H100** | sm_100 fallback, compile disabled |
| B300 SXM6 — Nightly 2.12+cu130 | 2.00 | **−67% vs H100** | sm_100 fallback, compile disabled |
| **B300 SXM6 — Nightly + torch.compile** | **2.74** | **−54% vs H100** | sm_100 fallback, compile working ✅ |

**With PyTorch Nightly + torch.compile, the gap narrows from 3–4× to ~2.2×** — a significant improvement, but the gap remains because sm_103 native cubins are still absent.

> **This is entirely a software stack problem, not a hardware deficiency.** With native sm_103 cubins and a fully working torch.compile path, B300 should exceed H100 by 1.2–1.5×.

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

#### Cause 2 — torch.compile / Triton Cannot Target sm_103 (Partially Fixed in Nightly)

On the H100 and B200 benchmarks that achieved 6–7 samples/sec, **torch.compile was almost certainly active and working**. Triton on H100/B200 compiles JIT kernels tuned to the exact GPU (sm_90 / sm_100), providing 10–30% additional speedup over PyTorch's precompiled cubins.

On B300 with **NGC 25.03** (PyTorch 2.7+CUDA 12.8): ptxas **rejects sm_103 entirely** (exit code 255), crashing all ranks. Dynamo is disabled entirely with `TORCHDYNAMO_DISABLE=1` — fully eager mode.

On B300 with **PyTorch Nightly 2.12+cu130**: torch.compile **works** — Dynamo traces the model, falls back to sm_100 compiled kernels, and delivers **+18–37% speedup** over eager mode (BF16: 2.00→2.74, FP16: 2.30→2.72 s/s). One function in the 3D Swin attention (`pangu.py:1045`) hits the recompile limit due to dynamic shapes and falls back to eager, but overall throughput still improves significantly.

**The key difference vs H100/B200:** H100/B200 torch.compile generates sm_90/sm_100 kernels *natively tuned to the hardware*. B300 nightly compile uses sm_100 cubins as a fallback — functional but not yet hardware-optimal. When native sm_103 cubins land, the +37% compile gain will stack on top of the cubin improvement.

#### Cause 3 — FusedAdam / Optimizer Not Available in NGC Container

H100/B200 reference runs used NVIDIA apex **FusedAdam**, which fuses all Adam parameter updates into a single CUDA kernel pass. The NGC 25.03 container does not include apex, so all NGC runs use PyTorch native AdamW (multiple kernel launches per parameter). This adds ~5–15% optimizer step overhead at scale.

### 8.4 Summary of the Gap — Current Status

| Source of gap | Estimated impact | conda | NGC | **Nightly** |
|---|---|---|---|---|
| sm_103 not in PyTorch arch list (sm_100 fallback) | 20–35% throughput loss | ❌ | ❌ | ❌ (still missing) |
| torch.compile / Triton not working | 10–30% throughput loss | Partial | ❌ broken | ✅ **Fixed (+18–37%)** |
| apex FusedAdam not in container | 5–15% throughput loss | ✅ (conda has apex) | ❌ | ❌ |
| **Remaining gap vs H100 (BF16, 4 GPU)** | | **~3.8×** | **~3.75×** | **~2.2×** |

**Nightly closes ~40% of the original gap** (from ~3.8× to ~2.2× vs H100) primarily through working torch.compile and improved cuBLAS 13.0 kernels. The remaining gap is almost entirely Cause 1 — sm_103 native cubins.

### 8.5 When Will B300 Match or Exceed H100/B200?

| Milestone | Expected Gain | Status (2026-03-17) |
|---|---|---|
| torch.compile working on B300 | +18–37% over eager | ✅ **Done — use Nightly** |
| PyTorch Nightly cuBLAS 13.0 improvements | +25–60% over NGC | ✅ **Done — Nightly delivers** |
| PyTorch release with native sm_103 cubins | +20–35% GEMM/Attention on top | ⏳ Not in `dev20260316`; watch arch_list |
| Triton JIT kernels native-tuned to sm_103 | +10–20% additional compile gain | ⏳ Blocked on sm_103 in arch_list |
| apex or FusedAdam in nightly env | +5–15% optimizer step | ⏳ Not installed; add manually |
| **Full stack maturity (all above)** | **B300 ~1.2–1.5× faster than H100** | ⏳ Estimated 2026 Q2–Q3 |

**Current best:** B300 Nightly + torch.compile = **2.74 s/s BF16** vs H100 ~6.0 s/s — **gap is ~2.2×**, down from the original ~3.8×. Once native sm_103 cubins land in nightly and Triton JIT targets sm_103, B300 should close to within H100 range and eventually exceed it.

---

## 9. Code Improvements & Tweaks Applied

### 9.1 `gpu_benchmark_dsai.py`

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

### 9.2 `exp1_dsai.yaml`

| Setting | Before | After | Reason |
|---|---|---|---|
| `optimizer_type` | `'Adam'` | `'FusedAdam'` | Apex FusedAdam is ~10% faster than native Adam in conda |
| `checkpointing` | `2` | `0` | B300 has 287 GB VRAM; gradient checkpointing is unnecessary |
| `torch_compile` | `False` | `True` | Enable torch.compile (works in conda env) |

### 9.3 `b300_training_dsai.sh`

- Removed `export NVIDIA_TF32_OVERRIDE=1` — this was silently overriding `--no-tf32` at the driver level, making FP32-pure runs actually execute with TF32, invalidating that test.
- Added NCCL tuning for Blackwell NVLink 5: `NCCL_BUFFSIZE=16777216`, `NCCL_MAX_NCHANNELS=32`.

### 9.4 `b300_training_ngc.sh` (new)

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

### 10.1 sm_103 Not in PyTorch's Compiled Arch List

**Problem:** NVIDIA B300 uses CUDA compute capability **SM 10.3 (sm_103)**. As of PyTorch 2.10, the officially compiled cubin list ends at:

```
sm_50, sm_60, sm_70, sm_80, sm_86, sm_90, sm_100
```

`sm_103` is absent. PyTorch falls back to `sm_100` cubins via binary compatibility, but these cubins lack sm_103-specific kernel tuning (Blackwell Tensor Core layouts, HBM3e access patterns). This is the primary reason GEMM and Attention benchmarks are ~20–35% below theoretical B300 peaks.

**Impact:** GEMM BF16 measures 1480–1560 TFLOPS vs theoretical 1750–2400 TFLOPS for Blackwell.

**Expected fix:** A future PyTorch release (likely 2.11 or 2.12) will add `sm_103` to the compiled arch list, providing native Blackwell cubin optimization.

### 10.2 torch.compile / Triton Failures

**Conda PyTorch 2.10 + CUDA 13.0:**
`torch.compile` calls Triton, which calls `ptxas --gpu-name=sm_103`. CUDA 13.0's `ptxas` supports sm_103 and compiles, but the resulting kernels may not be fully optimized.

**NGC 25.03 PyTorch 2.7 + CUDA 12.8:**
CUDA 12.8's `ptxas` returns **exit code 255** for `--gpu-name=sm_103` — sm_103 is not recognized. This causes Triton / torch.compile to crash all distributed ranks at the first compiled kernel invocation.

**Fix applied:** `TORCHDYNAMO_DISABLE=1` prevents dynamo graph capture, forcing eager mode. The trainer still logs "torch.compile enabled" (the wrapper is applied) but actual JIT compilation is skipped.

**Implication:** All NGC 25.03 results are **eager mode** — no kernel fusion from torch.compile. The FP16 speedup (65% over conda) comes from PyTorch 2.7's improved eager kernels and cuBLAS 12.8, not compilation.

### 10.3 "Using PyTorch native" — What It Means

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

### 10.4 HBM3e Bandwidth Saturation

Measured D2D bandwidth via `tensor.copy_()`: **2971 GB/s** vs HBM3e theoretical **~8 TB/s** (~37% utilization).

This is a software limitation, not a hardware failure. The `tensor.copy_()` code path:
1. Uses a single CUDA stream
2. Goes through PyTorch's memory management layer
3. Cannot directly address all HBM3e memory controllers simultaneously

Achieving 7–8 TB/s requires raw `cudaMemcpy` with explicit multi-controller parallelism — below PyTorch's abstraction level.

Training workloads (GEMM-dominant) are compute-bound, not memory-bound, so this does not impact training throughput in practice.

### 10.5 FP8 / Transformer Engine

FP8 via NVIDIA Transformer Engine (`--amp-dtype fp8`) showed **no speedup** over BF16 (1.60 vs 1.60 samples/sec in NGC). This is expected:

FP8 only accelerates layers replaced by `transformer_engine.pytorch.Linear`. The Pangu model uses standard `torch.nn.Linear` throughout. Without replacing the layer definitions, TE operates in a "compatibility" mode that adds FP8 metadata overhead without delivering the actual Tensor Core speedup.

Full FP8 benefit requires replacing `nn.Linear` with `te.Linear` throughout the Pangu architecture — a non-trivial model change.

### 10.6 Docker Group Permission

Initial Docker runs failed with `permission denied`. Fixed by adding the user to the docker group:

```bash
sudo usermod -aG docker rdesouz4
# For current session (without re-login):
sg docker -c "docker run ..."
```

---

## 11. PyTorch Nightly + CUDA 13.0 Benchmark

**Script:** `b300_training_nightly.sh` — self-bootstrapping, creates conda env `pt-nightly-cu130` on first run.
**Motivation:** Based on Gemini/PyTorch upstream status (2026-03-17), nightly cu130 builds are the first to include experimental native **sm_103 cubins** and a patched Triton that generates sm_103 PTX — directly targeting the two primary causes of the 3–4× performance gap identified in §8.

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash training/scripts/b300_training_nightly.sh \
    2>&1 | tee nightly_results.log
```

### 11.1 New Capabilities Tested (vs conda / NGC)

| Feature | Conda 2.10+cu130 | NGC 2.7+cu128 | **Nightly cu130** |
|---|---|---|---|
| sm_103 native cubins | No (sm_100 fallback) | No | **No** (sm_100 fallback; sm_120 present but ≠ sm_103) |
| torch.compile / Triton | Partial | Broken (ptxas 255) | **Working** (sm_100 fallback; +18–37% speedup) |
| FP8 Tensor Core throughput | Partial (no te.Linear) | Partial | **Better** (2.55 s/s, fastest precision) |
| **FP4 via torchao** | No | No | API available; torchao nightly required (0.16.0 incompatible) |

**FP4 context:** B300 Tensor Cores support hardware-accelerated dense FP4 at up to **14 PFLOPS** — 3× the FP8 rate (4.5 PFLOPS) and 6× BF16 (2.4 PFLOPS). CUTLASS 4.4 added explicit sm_103 block-scaled FP4 GEMM support. `torchao` exposes this from Python via `fp4_weight_only()` quantization.

### 11.2 sm_103 Validation

The script probes `torch.cuda.get_arch_list()` and `ptxas --gpu-name=sm_103` at startup, reporting:

```
sm_103 native:  YES / NO
ptxas sm_103:   OK / FAIL (exit 255)
```

This directly answers whether the nightly build closes the cubin gap from §8.3 Cause 1.

### 11.3 Precision Sweep Results (4 GPU, batch_size=8)

**PyTorch `2.12.0.dev20260316+cu130` — torch.compile enabled (mode=reduce-overhead)**

| Precision | Nightly (s/s) | Conda 2.10 (s/s) | NGC 25.03 (s/s) | vs Conda | vs NGC |
|---|---|---|---|---|---|
| BF16 | **2.00** | 1.58 | 1.60 | **+27%** | **+25%** |
| FP16 | **2.30** | 1.00 | 1.65 | **+130%** | **+39%** |
| FP32 + TF32 | **2.45** | 0.94 | 1.26 | **+161%** | **+94%** |
| FP8 (TE) | **2.55** | 1.12 | 1.60 | **+128%** | **+59%** |

**Finding:** Nightly significantly outperforms both conda and NGC across all precisions. FP32+TF32 shows the largest relative gain (+94% over NGC), and FP8 is now the fastest precision at 2.55 s/s — suggesting Transformer Engine integration is improving with newer PyTorch.

### 11.4 Batch Size Sweep (4 GPU, FP16)

| Batch | Nightly (s/s) | Conda (s/s) | NGC 25.03 (s/s) | vs Conda | vs NGC |
|---|---|---|---|---|---|
| 4 | **1.15** | 0.76 | 1.37 | +51% | -16% |
| 8 | **2.63** | 1.01 | 1.55 | **+160%** | **+70%** |
| 12 | **2.81** | 1.17 | 1.84 | **+140%** | **+53%** |

> Note: batch=4 nightly (1.15) is lower than NGC (1.37) — likely due to torch.compile warmup overhead at small batch sizes amortizing poorly over only 50 steps.

### 11.5 GPU Weak Scaling (per-GPU batch=2, FP16)

| GPUs | Nightly (s/s) | Conda (s/s) | NGC 25.03 (s/s) | Scaling efficiency |
|---|---|---|---|---|
| 1 | **0.85** | 0.31 | 0.59 | 100% (baseline) |
| 2 | **1.50** | 0.57 | 1.07 | 88.2% |
| 4 | **2.71** | 1.12 | 1.55 | 79.7% |

### 11.6 torch.compile Sweep — Task 6 (nightly only) ⭐

> **Key finding: torch.compile works in nightly and provides significant speedup on B300.**
> Unlike NGC 25.03 (ptxas crash) and conda (partial), nightly Dynamo traces successfully
> and falls back to sm_100 compiled kernels — still faster than pure eager mode.

| Precision | Eager (s/s) | + compile (s/s) | Compile gain |
|---|---|---|---|
| BF16 | 2.00 | **2.74** | **+37%** |
| FP16 | 2.30 | **2.72** | **+18%** |

**torch.compile behavior in nightly:**
- Dynamo traces the model successfully (`mode=reduce-overhead`)
- Hits recompile limit (8) on `pangu.py:1045` due to dynamic shapes in the 3D Swin attention window — falls back to eager for that function only
- Despite partial compilation, wall-clock throughput improves significantly (+18–37%)
- This is the strongest evidence yet that native sm_103 cubins (when they arrive) will unlock the full B300 performance potential

### 11.7 FP4 via torchao — Task 7

FP4 benchmark **could not run** — `torchao 0.16.0` API is incompatible with `PyTorch 2.12.0.dev20260316`:

```
FP4 test failed: cannot import name 'float8_dynamic_activation_float8_weight'
from 'torchao.quantization'
Note: torchao 0.16.0 cpp extensions skipped due to incompatible torch version 2.12.0.dev20260316
```

**Root cause:** torchao 0.16.0 was released targeting stable PyTorch 2.5/2.6 APIs that changed in the nightly. Fix: install torchao nightly alongside PyTorch nightly:
```bash
pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu130
```

### 11.8 Summary — Three-Way Comparison

| Metric | Conda 2.10 | NGC 25.03 | **Nightly 2.12** | Best |
|---|---|---|---|---|
| BF16 peak (s/s) | 1.58 | 1.60 | **2.00** | Nightly +27% |
| FP16 peak (s/s) | 1.00 | 1.65 | **2.30** | Nightly +39% over NGC |
| FP8 (s/s) | 1.12 | 1.60 | **2.55** | Nightly +59% over NGC |
| BF16 + compile (s/s) | N/A | N/A (crash) | **2.74** | Nightly only |
| sm_103 native cubins | No | No | No | None yet |
| torch.compile | Partial | Broken | **Working (+37%)** | Nightly |
| NCCL version | 2.28.9 | 2.25.1 | **2.29.3** | Nightly |

**Verdict on nightly:** Even without native sm_103 cubins, PyTorch 2.12 nightly delivers **+25–60% throughput** over NGC 25.03 through improved cuBLAS 13.0 kernels, better DDP communication overlap, and a working torch.compile path. **This is now the recommended stack for B300.**

---

## 12. MLPerf Inference — ResNet-50 & BERT-Large Offline

**Scripts:** `mlperf/mlperf_b300_resnet50.py`, `mlperf/mlperf_b300_bert.py`
**Runner:** `mlperf/run_mlperf_b300.sh` (conda) · `mlperf/run_mlperf_b300_docker.sh` (NGC Docker)
**Scenario:** MLPerf Offline, PerformanceOnly mode, synthetic data (no dataset download)
**GPUs:** 4× B300 SXM6 AC (`CUDA_VISIBLE_DEVICES=4,5,6,7`)

### 12.1 Setup & Design

Both benchmarks use `torch.nn.DataParallel` across 4 GPUs with FP16 precision. The dataset is
kept on CPU and scattered per batch to support DataParallel's scatter/gather pattern.
`mlperf_loadgen` is built from source (`mlcommons/inference`) since no sm_103-compatible
binary is available on PyPI.

**Key fix — response pointer:** Python 3.9+ breaks `memoryview.__array_interface__`; fixed
by using `buf.ctypes.data` for the raw C pointer passed to `QuerySampleResponse`.

| Setting | ResNet-50 | BERT-Large |
|---|---|---|
| Model | ResNet-50 V2 (IMAGENET1K_V2) | BertModel (hidden=1024, 24 layers) |
| Precision | FP16 | FP16 |
| Batch size (4 GPU) | 256 (64 per GPU) | 128 (32 per GPU) |
| Sequence length | — | 384 (MLPerf standard) |
| Synthetic samples | 5,000 | 2,000 |
| min_duration_ms | 60,000 | 60,000 |

### 12.2 Results (4× B300 SXM6, FP16, synthetic data)

| Model | QPS (measured) | Batch | Result | Notes |
|---|---|---|---|---|
| ResNet-50 FP16 Offline | **405.6 QPS** | 256 | VALID | DataParallel, 4× B300 |
| BERT-Large FP16 Offline | **1524.8 QPS** | 128 | VALID | DataParallel, seq_len=384 |

### 12.3 Notes on B300 MLPerf Performance

- **No native sm_103 cubins** — same software gap as training benchmarks; inference kernels
  use sm_100 fallback, expect 20–35% below theoretical peak throughput.
- **DataParallel vs DistributedDataParallel** — DataParallel is used here for simplicity
  (single-process, multi-GPU scatter). For production MLPerf submissions, each GPU would
  run its own SUT process with a shared queue, eliminating Python GIL contention.
- **Synthetic data** — results reflect pure GPU throughput with no I/O or preprocessing
  bottleneck. Real dataset runs may differ if tokenization/decoding becomes a bottleneck.
- **Target QPS setting** — `offline_expected_qps` is set conservatively (3,000/GPU for
  ResNet-50, 800/GPU for BERT) to avoid loadgen issuing millions of samples in one query
  (which caused hours-long Python iteration loops at the original 30,000/GPU target).

---

## 13. Verdict & Recommendations

### Overall Performance Summary

| Environment | BF16 (s/s) | FP16 (s/s) | FP32+TF32 (s/s) | FP8 (s/s) | Recommendation |
|---|---|---|---|---|---|
| Conda PyTorch 2.10+cu130 | 1.58 | 1.00 | 0.94 | 1.12 | BF16 only; baseline |
| NGC 25.03 PyTorch 2.7+cu128 | 1.60 | 1.65 | 1.26 | 1.60 | All precision modes |
| **Nightly 2.12.0.dev+cu130** | **2.00** | **2.30** | **2.45** | **2.55** | **Recommended — fastest across all modes** |
| Nightly + torch.compile | **2.74** | **2.72** | — | — | **Peak throughput** |

### Verdict

**Use PyTorch Nightly cu130 (`pt-nightly-cu130` env) for all B300 training.** Reasons:

1. **BF16 +25% over NGC** — the most common training precision gains significantly
2. **FP16 +39% over NGC** — nightly is now clearly faster across every precision
3. **FP8 is now the fastest precision** — 2.55 s/s vs 1.60 NGC; Transformer Engine improving
4. **torch.compile works** — unlike NGC (ptxas crash); +18–37% additional speedup over eager
5. **Better NCCL** — 2.29.3+cuda13.1 vs NGC's 2.25.1; better DDP communication overlap
6. **Self-bootstrapping script** — `b300_training_nightly.sh` creates the conda env on first run

> **Fallback:** If nightly instability is a concern (e.g., production runs), use NGC 25.03 as the stable alternative — it still delivers +65% FP16 over conda.

### What to Expect Next (Future Improvements)

| Improvement | Expected Gain | Status |
|---|---|---|
| PyTorch with native sm_103 cubins | +20–35% GEMM/Attention | Not yet in `dev20260316`; watch for sm_103 in arch_list |
| apex FusedAdam in NGC / nightly container | +5–15% optimizer step | Add `apex` to `Dockerfile.ngc` |
| Full FP8 (`te.Linear` in model) | +30–50% vs BF16 | Requires model architecture surgery |
| torch.compile on B300 | ✅ Working in nightly (+18–37%) | NGC still broken; use nightly |
| torchao FP4 on B300 | 14 PFLOPS potential | Install torchao nightly alongside PyTorch nightly |

### Quick Start

**Option A — Nightly (recommended):**
```bash
# From repo root — self-bootstrapping, creates pt-nightly-cu130 env if missing
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/scripts/b300_training_nightly.sh \
    2>&1 | tee b300_results_nightly.log
```

**Option B — NGC Container (stable fallback):**
```bash
# Build image once (auto-triggered by script if missing)
sg docker -c "docker build -f Dockerfile.ngc -t pangu-s2s-ngc:latest ."

# Run full benchmark suite
NGC_GPUS=0,1,2,3 bash training/scripts/b300_training_ngc.sh \
    2>&1 | tee b300_results_ngc.log
```

---

*Report generated from benchmarks run on 2026-03-16/17 on node b301.*
*All results: 50 training steps, Pangu S2S 79M parameters, 4× NVIDIA B300 SXM6 AC.*
*PyTorch Nightly: `2.12.0.dev20260316+cu130`. MLPerf Inference: synthetic data, offline scenario.*
