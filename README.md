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
│   │   └── b300_training_ngc.sh      # Benchmark suite — NGC container (recommended)
│   ├── networks/
│   │   ├── pangu.py                  # 3D Swin Transformer architecture
│   │   └── pangu_lite.py
│   └── utils/                        # Data loaders, losses, metrics, etc.
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
| sm_103 absent from PyTorch cubin list | GEMM/Attn 20–35% below peak | PyTorch 2.11/2.12 |
| Triton/ptxas rejects sm_103 in CUDA 12.8 | torch.compile crashes in NGC | CUDA 12.9+ or Triton patch |
| apex FusedAdam not in NGC container | ~5–15% optimizer overhead | Add to Dockerfile.ngc |
| FP8 only partial (no te.Linear in model) | No FP8 speedup | Requires model surgery |

See [`report/b300_benchmark_report.md`](report/b300_benchmark_report.md) §8–10 for full analysis.

---

## Citation / References

- Pangu-Weather S2S: [NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo/examples/weather/pangu_weather/README.html)
- NCCL Tests: https://github.com/NVIDIA/nccl-tests
- NGC PyTorch 25.03: `nvcr.io/nvidia/pytorch:25.03-py3`
