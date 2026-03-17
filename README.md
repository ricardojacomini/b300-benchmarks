# B300 Benchmark Suite — Reproducible Tests

Two reproducible benchmarks for the **NVIDIA B300 SXM6 (Blackwell, SM 10.3)**:

1. **GPU Microbenchmarks** — GEMM, Attention, Conv2D, Memory BW, NVLink all-reduce
2. **Pangu S2S Training** — end-to-end ML training benchmark (conda and NGC container)

Results, findings, and root-cause analysis are in [`report/b300_benchmark_report.md`](report/b300_benchmark_report.md).

---

## Repository Layout

```
b300-benchmarks/
├── Dockerfile.ngc                    # NGC 25.03 image with all Pangu deps
├── benchmarks/
│   └── gpu_benchmark_dsai.py         # GPU microbenchmark (GEMM/Attn/Conv/BW/NCCL)
├── training/
│   ├── faster_train.py               # Pangu S2S trainer (DDP, AMP, compile)
│   ├── config/
│   │   └── exp1_dsai.yaml            # Model & training hyperparameters
│   ├── scripts/
│   │   ├── b300_training_dsai.sh     # Benchmark suite — conda environment
│   │   ├── b300_training_ngc.sh      # Benchmark suite — NGC container (recommended)
│   │   └── b300_training_nightly.sh  # Benchmark suite — PyTorch nightly cu130 (bleeding-edge)
│   ├── networks/
│   │   ├── pangu.py                  # 3D Swin Transformer architecture
│   │   └── pangu_lite.py
│   └── utils/                        # Data loaders, losses, metrics, etc.
├── mlperf/
│   ├── mlperf_b300_resnet50.py       # ResNet-50 FP16 offline SUT (DataParallel)
│   ├── mlperf_b300_bert.py           # BERT-Large FP16 offline SUT (DataParallel)
│   ├── run_mlperf_b300.sh            # Runner — conda s2s env
│   ├── run_mlperf_b300_docker.sh     # Runner — NGC Docker container
│   ├── Dockerfile.mlperf             # NGC 25.03 + mlperf_loadgen from source
│   └── README.md                     # MLPerf setup & results
├── dataset/
│   └── README.md                     # Dataset layout + access instructions
└── report/
    └── b300_benchmark_report.md      # Full benchmark report with tables & analysis
```

---

## Requirements

### Hardware
- NVIDIA B300 SXM6 (or any Blackwell / Hopper / Ampere GPU)
- At least 4 GPUs for multi-GPU tests (single-GPU also supported)
- NVLink recommended for collective bandwidth tests

### Software — Option A: Conda (baseline)
```bash
module load anaconda3/2024.02-1
conda create -n s2s python=3.11 -y
conda activate /scratch/$USER/envs/s2s

pip install torch==2.10.0+cu130 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130

pip install wandb einops timm ruamel.yaml xarray h5py \
    cartopy cftime pandas psutil tqdm netcdf4 h5netcdf \
    nvidia-nccl-cu13

# Optional but recommended for FusedAdam:
pip install apex
```

### Software — Option C: PyTorch Nightly + CUDA 13.0 (bleeding-edge B300 support)

```bash
conda create -n pt-nightly-cu130 python=3.11 -y
conda activate /scratch/$USER/envs/pt-nightly-cu130

pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130

pip install wandb einops timm ruamel.yaml xarray h5py \
    cartopy cftime pandas psutil tqdm netcdf4 h5netcdf torchao
```

> **Exact environment lock** (captured 2026-03-17, `torch==2.12.0.dev20260316+cu130`):
> ```bash
> pip install -r requirements-nightly-cu130.txt
> ```
> See [`requirements-nightly-cu130.txt`](requirements-nightly-cu130.txt) to reproduce the exact environment used for the §11 benchmark results.

Run the nightly benchmark suite (self-bootstrapping, creates env if missing):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash training/scripts/b300_training_nightly.sh \
    2>&1 | tee nightly_results.log
```

#### Why use PyTorch Nightly?

> *Nightly builds are the development snapshots published every day from the main PyTorch branch. They are not stable — APIs can change, bugs can appear — but they serve three important purposes on new hardware like the B300:*

**1. Bleeding-Edge Hardware Support**
As seen with the NVIDIA B300 (sm_103), official support takes time to reach a stable release.
Nightly builds are often the only way to use brand-new architectures or specific CUDA versions
(like CUDA 13.0+). The nightly track is where sm_103 cubins and FP4 Tensor Core support
will appear first — months before the next stable PyTorch release.

**2. New Features First**
The absolute latest performance optimizations in `torch.compile`, new data types like
hardware-accelerated FP4 (B300's flagship: 14 PFLOPS), and new quantization routines
in `torchao` show up in Nightly builds months before they reach a stable version.

**3. Immediate Bug Fixes**
If you hit a critical bug in a stable release and a developer fixes it on GitHub,
you can install the next day's Nightly build to get the fix immediately rather than
waiting months for the next stable release.

#### The Verdict: Should You Use It?

| Use Case | Recommendation |
|---|---|
| **Production / serving** | ❌ Use stable releases. Predictability is paramount. |
| **R&D on new hardware (B300, B200 Ultra)** | ✅ Nightly is standard practice. |
| **Testing new quantization (FP4, FP8)** | ✅ Features arrive here first. |
| **torch.compile research** | ✅ Fastest path to sm_103 kernel support. |

> **Important:** Pin your nightly environment (e.g., record the exact build date in a
> `requirements.txt` or lock it in a Docker image) so you can roll back if tomorrow's
> build breaks your script.

#### B300 Nightly Status (as of 2026-03-17, build `2.12.0.dev20260316+cu130`)

| Feature | Status |
|---|---|
| sm_103 in `torch.cuda.get_arch_list()` | ❌ Not yet (`sm_120` present but not `sm_103`) |
| ptxas sm_103 support | ❌ Still exits 255 (CUDA 13.0 not enough) |
| torch.compile on B300 | ❌ Falls back (Triton cannot target sm_103) |
| FP4 via torchao | ✅ API available; hw acceleration pending sm_103 cubins |
| Perf vs conda stable | Similar to NGC 25.03 (same root cause: sm_100 fallback) |

> **Implication:** Gemini's statement that "nightly cu130 has sm_103 support" is
> directionally correct but the specific March 2026 nightly (`dev20260316`) still
> uses `sm_100` cubins as fallback. Watch for the build that adds `sm_103` to the
> arch list — that will be the inflection point.

---

### Software — Option B: NGC Container (recommended for B300)
```bash
# Pull once
docker pull nvcr.io/nvidia/pytorch:25.03-py3

# Build custom image with Pangu deps pre-installed
docker build -f Dockerfile.ngc -t pangu-s2s-ngc:latest .
```

> **Why NGC 25.03?**
> PyTorch 2.7 + CUDA 12.8 in NGC 25.03 delivers **+65% FP16 throughput** over
> the conda baseline on B300. See the report for detailed analysis.

---

## Test 1 — GPU Microbenchmarks

**Script:** `benchmarks/gpu_benchmark_dsai.py`

### Conda
```bash
# Set CONDA_ENV so the script resolves the NCCL library path
export CONDA_ENV=/scratch/$USER/envs/s2s
export LD_LIBRARY_PATH=${CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

# 4-GPU run on GPUs 4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7 python benchmarks/gpu_benchmark_dsai.py \
    --output results/gpu_benchmark.yaml 2>&1 | tee results/gpu_benchmark.log
```

### NGC Container
```bash
docker run --rm --gpus '"device=4,5,6,7"' --ipc=host \
    -v $(pwd):/workspace -w /workspace \
    pangu-s2s-ngc:latest \
    python benchmarks/gpu_benchmark_dsai.py \
        --output /workspace/results/gpu_benchmark_ngc.yaml
```

### Expected output (B300, 4 GPU)
| Test | Measured | Notes |
|---|---|---|
| GEMM BF16 (4096³) | ~1558 TFLOPS | sm_100 fallback; expect +20% with native sm_103 cubins |
| Attention FP16 (B8 H40 S4096) | ~126 TFLOPS | |
| D2D Memory BW | ~2971 GB/s | PyTorch cap; HBM3e theoretical ~8 TB/s |
| All-Reduce 1 GB | ~654 GB/s bus BW | 72.7% of NVLink 5 theoretical |

---

## Test 2 — Pangu S2S Training

**Dataset required:** See [`dataset/README.md`](dataset/README.md) for layout and access.

The training config reads data from the path set in `training/config/exp1_dsai.yaml`:
```yaml
data_dir: '/path/to/dataset'   # update this to your dataset location
```

Or override at runtime:
```bash
# conda
--data-dir-override /path/to/your/dataset

# NGC
DATASET_DIR=/path/to/your/dataset bash training/scripts/b300_training_ngc.sh
```

### Option A — Conda (baseline, BF16 ≈ 1.58 samples/sec)
```bash
# From repo root
cd training/scripts
bash b300_training_dsai.sh 2>&1 | tee b300_results_conda.log
```

Runs all 5 benchmark tasks (precision sweep, batch sweep, data location,
strong scaling, weak scaling) and prints a summary table at the end.

### Option B — NGC Container (recommended, FP16 ≈ 1.65 samples/sec)
```bash
# From repo root — self-bootstrapping, builds image if missing
NGC_GPUS=4,5,6,7 bash training/scripts/b300_training_ngc.sh \
    2>&1 | tee b300_results_ngc.log
```

The script detects whether it's on the host or inside Docker and
re-launches itself in the container automatically.

### Key environment variables
| Variable | Default | Description |
|---|---|---|
| `NGC_GPUS` | `4,5,6,7` | GPU indices to use |
| `DATASET_DIR` | `<repo_root>/dataset` | Path to ERA5 dataset |
| `RUN_NUM` | `300` | Starting run number (checkpoint dir) |

### Expected results (4× B300, batch_size=8)
| Precision | Conda samples/sec | NGC samples/sec |
|---|---|---|
| BF16 | 1.58 | 1.60 |
| FP16 | 1.00 | **1.65** |
| FP32 + TF32 | 0.94 | 1.26 |
| FP32 pure | 0.98 | 1.20 |
| FP8 / TE | 1.12 | 1.60 |

---

## Test 3 — MLPerf Inference (ResNet-50 + BERT-Large Offline)

**Scripts:** `mlperf/` — see [`mlperf/README.md`](mlperf/README.md) for full details.

No dataset download required — uses synthetic data for throughput measurement.

### Prerequisites — build mlperf_loadgen from source

```bash
git clone --depth=1 https://github.com/mlcommons/inference.git /tmp/mlperf-inference
cd /tmp/mlperf-inference/loadgen
CFLAGS="-std=c++14" pip install --no-cache-dir .
```

### Run (conda, 4 GPUs)

```bash
cd mlperf/
CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_mlperf_b300.sh
```

### Run (NGC Docker)

```bash
cd mlperf/
docker build -f Dockerfile.mlperf -t mlperf-b300-ngc:latest .
bash run_mlperf_b300_docker.sh
```

### Expected results (4× B300 SXM6, FP16, synthetic data)

| Model | Batch | QPS | Result | Scenario |
|---|---|---|---|---|
| ResNet-50 FP16 | 256 (64/GPU) | **405.6** | VALID | Offline |
| BERT-Large FP16 (seq=384) | 128 (32/GPU) | **1524.8** | VALID | Offline |

---

## NCCL Tests (Optional — raw NVLink bandwidth)

```bash
# Clone and build (one time)
git clone https://github.com/NVIDIA/nccl-tests /tmp/nccl-tests

NCCL_LIB=/scratch/$USER/envs/s2s/lib/python3.11/site-packages/nvidia/nccl/lib
ln -sf ${NCCL_LIB}/libnccl.so.2 ${NCCL_LIB}/libnccl.so

make -C /tmp/nccl-tests MPI=1 \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/scratch/$USER/envs/s2s/lib/python3.11/site-packages/nvidia/nccl \
    MPI_HOME=/usr/mpi/gcc/openmpi-4.1.9a1

# Run all-reduce test (4 GPUs, 1 MB → 1 GB)
mpirun -np 4 /tmp/nccl-tests/build/all_reduce_perf -b 1M -e 1G -f 2 -g 1
```

Expected peak bus bandwidth on B300: **~654 GB/s** at 1 GB (72.7% of NVLink 5).

---

## Known Limitations on B300 (as of 2026-03)

| Issue | Impact | Fix ETA |
|---|---|---|
| sm_103 absent from PyTorch cubin list | GEMM/Attn 20–35% below peak | Not in nightly dev20260316 either; watch for sm_103 in arch_list |
| Triton/ptxas rejects sm_103 (CUDA 12.8 & 13.0) | torch.compile unusable on B300 | Requires upstream Triton patch + ptxas sm_103 support |
| apex FusedAdam not in NGC/nightly container | ~5–15% optimizer overhead | Add to Dockerfile.ngc |
| FP8 only partial (no te.Linear in model) | No FP8 speedup | Requires model architecture surgery |
| FP4 (torchao) API available but no hw acceleration | 14 PFLOPS unreachable without sm_103 cubins | Will unlock once sm_103 lands in nightly |

See [`report/b300_benchmark_report.md`](report/b300_benchmark_report.md) §8–10 for full analysis.

---

## Citation / References

- Pangu-Weather S2S: [NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo/examples/weather/pangu_weather/README.html)
- NCCL Tests: https://github.com/NVIDIA/nccl-tests
- NGC PyTorch 25.03: `nvcr.io/nvidia/pytorch:25.03-py3`
- **PyTorch Nightly Builds** — updated every night from the `main` branch:
  - Install index: https://download.pytorch.org/whl/nightly/cu130
  - Release notes (stable + nightly): https://github.com/pytorch/pytorch/releases
  - Nightly release tracker (check which `sm_*` archs are added): https://github.com/pytorch/pytorch/issues
  - Used in this report: `torch==2.12.0.dev20260316+cu130` (captured 2026-03-17)
  - Lock file: [`requirements-nightly-cu130.txt`](requirements-nightly-cu130.txt)
