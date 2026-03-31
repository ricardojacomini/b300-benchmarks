#!/usr/bin/env python3
"""
NVIDIA GPU ML Benchmark Suite
Comprehensive benchmarking for A100, H100, L40S, B200/B300 architectures
Tests: GEMM, Conv2D, Attention, Mixed Precision, Tensor Cores, Memory Bandwidth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import json
import yaml
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import sys
import os

# Set NCCL library path from conda environment
#CONDA_ENV = "/scratch/rdesouz4/envs/s2s"
#os.environ["LD_LIBRARY_PATH"] = (
#    f"{CONDA_ENV}/lib/python3.11/site-packages/nvidia/nccl/lib"
#    + (":" + os.environ["LD_LIBRARY_PATH"] if "LD_LIBRARY_PATH" in os.environ else "")
#)

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    gpu_name: str
    gpu_memory_gb: float
    cuda_capability: str
    test_name: str
    operation: str
    dtype: str
    shape: str
    throughput_tflops: float
    time_ms: float
    memory_bandwidth_gb_s: float = 0.0

class GPUBenchmark:
    def __init__(self, device_id: int = 0, multi_gpu: bool = False, rank: int = 0, world_size: int = 1):
        self.device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(self.device)
        self.results: List[BenchmarkResult] = []
        self.multi_gpu = multi_gpu
        self.rank = rank
        self.world_size = world_size
        
        # GPU info
        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        self.cuda_capability = f"{torch.cuda.get_device_properties(self.device).major}.{torch.cuda.get_device_properties(self.device).minor}"
        
        # Enable cuDNN autotuning and TF32 for best conv performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        if rank == 0:  # Only print from rank 0
            print(f"\n{'='*70}")
            if multi_gpu:
                print(f"Multi-GPU Mode: {world_size} GPUs")
                print(f"Rank 0 GPU: {self.gpu_name}")
            else:
                print(f"GPU: {self.gpu_name}")
            print(f"Memory: {self.gpu_memory:.2f} GB")
            print(f"CUDA Capability: {self.cuda_capability}")
            print(f"{'='*70}\n")
        
    def warmup(self, iterations: int = 10):
        """Warmup GPU"""
        if self.rank == 0:
            print("Warming up GPU...")
        x = torch.randn(1024, 1024, device=self.device)
        for _ in range(iterations):
            _ = torch.matmul(x, x)
        torch.cuda.synchronize()
        if self.multi_gpu:
            dist.barrier()
        if self.rank == 0:
            print("Warmup complete.\n")
    
    def benchmark_matmul(self, M: int, N: int, K: int, dtype: torch.dtype, iterations: int = 100):
        """Benchmark matrix multiplication (GEMM)"""
        A = torch.randn(M, K, dtype=dtype, device=self.device)
        B = torch.randn(K, N, dtype=dtype, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        
        # Calculate FLOPS: 2*M*N*K operations per matmul
        flops = 2 * M * N * K
        tflops = (flops / (time_ms / 1000)) / 1e12
        
        # Memory bandwidth (simplified)
        bytes_transferred = (M*K + K*N + M*N) * A.element_size()
        bandwidth = (bytes_transferred / (time_ms / 1000)) / 1e9
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="GEMM",
            operation=f"MatMul ({M}x{K}) @ ({K}x{N})",
            dtype=str(dtype).split('.')[-1],
            shape=f"{M}x{N}x{K}",
            throughput_tflops=tflops,
            time_ms=time_ms,
            memory_bandwidth_gb_s=bandwidth
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  MatMul {M}x{N}x{K} ({dtype}): {tflops:.2f} TFLOPS, {time_ms:.3f} ms")
        
        del A, B, C
        torch.cuda.empty_cache()
        
        return tflops
    
    def benchmark_conv2d(self, batch: int, channels_in: int, channels_out: int, 
                         height: int, width: int, kernel_size: int, dtype: torch.dtype, 
                         iterations: int = 100):
        """Benchmark 2D convolution"""
        x = torch.randn(batch, channels_in, height, width, dtype=dtype, device=self.device).to(memory_format=torch.channels_last)
        conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=kernel_size//2).to(dtype).to(self.device).to(memory_format=torch.channels_last)

        # Warmup — use more iterations so cuDNN benchmark mode can find the best algorithm
        for _ in range(30):
            _ = conv(x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            y = conv(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        
        # Approximate FLOPS for convolution
        output_size = height * width  # assuming same padding
        flops_per_output = 2 * channels_in * kernel_size * kernel_size
        total_flops = batch * channels_out * output_size * flops_per_output
        tflops = (total_flops / (time_ms / 1000)) / 1e12
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="Conv2D",
            operation=f"Conv2D B{batch}xC{channels_in}x{height}x{width} K{kernel_size}",
            dtype=str(dtype).split('.')[-1],
            shape=f"{batch}x{channels_in}x{height}x{width}",
            throughput_tflops=tflops,
            time_ms=time_ms
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  Conv2D {batch}x{channels_in}x{height}x{width} k{kernel_size} ({dtype}): {tflops:.2f} TFLOPS, {time_ms:.3f} ms")
        
        del x, conv, y
        torch.cuda.empty_cache()
        
        return tflops
    
    def benchmark_attention(self, batch: int, seq_len: int, heads: int, 
                           head_dim: int, dtype: torch.dtype, iterations: int = 50):
        """Benchmark multi-head attention (critical for transformers)"""
        Q = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        K = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        V = torch.randn(batch, heads, seq_len, head_dim, dtype=dtype, device=self.device)
        
        # Warmup
        for _ in range(5):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn, V)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, V)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        
        # FLOPS: 2 matmuls + softmax
        flops_qk = 2 * batch * heads * seq_len * seq_len * head_dim
        flops_av = 2 * batch * heads * seq_len * seq_len * head_dim
        total_flops = flops_qk + flops_av
        tflops = (total_flops / (time_ms / 1000)) / 1e12
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="Attention",
            operation=f"MHA B{batch} H{heads} S{seq_len} D{head_dim}",
            dtype=str(dtype).split('.')[-1],
            shape=f"{batch}x{heads}x{seq_len}x{head_dim}",
            throughput_tflops=tflops,
            time_ms=time_ms
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  Attention B{batch} H{heads} S{seq_len} D{head_dim} ({dtype}): {tflops:.2f} TFLOPS, {time_ms:.3f} ms")
        
        del Q, K, V, scores, attn, output
        torch.cuda.empty_cache()
        
        return tflops
    
    def benchmark_memory_bandwidth(self, size_mb: int = 1024, iterations: int = 100, num_streams: int = 8):
        """Benchmark memory bandwidth using concurrent streams to saturate all HBM memory controllers"""
        size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        # Align chunk size to 256-byte boundary for optimal memory controller access
        chunk = (size_elements // num_streams) & ~63  # align to 64 float32s = 256 bytes
        actual_elements = chunk * num_streams

        src = torch.randn(actual_elements, dtype=torch.float32, device=self.device)
        dst = torch.empty_like(src)
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]

        def parallel_copy():
            for i, s in enumerate(streams):
                with torch.cuda.stream(s):
                    dst[i*chunk:(i+1)*chunk].copy_(src[i*chunk:(i+1)*chunk], non_blocking=True)
            torch.cuda.synchronize(self.device)

        # Warmup
        for _ in range(10):
            parallel_copy()

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            parallel_copy()
        end = time.perf_counter()

        time_ms = (end - start) * 1000 / iterations
        bandwidth = (size_mb / 1024) / (time_ms / 1000)  # GB/s
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="Memory",
            operation=f"Device-to-Device Copy {size_mb} MB",
            dtype="float32",
            shape=f"{size_elements}",
            throughput_tflops=0.0,
            time_ms=time_ms,
            memory_bandwidth_gb_s=bandwidth
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  Device-to-Device Copy {size_mb} MB: {bandwidth:.2f} GB/s, {time_ms:.3f} ms")
        
        del src, dst, streams
        torch.cuda.empty_cache()

        return bandwidth

    def benchmark_host_to_device(self, size_mb: int = 1024, iterations: int = 100):
        """Benchmark host-to-device (CPU to GPU) transfer bandwidth"""
        size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        # Create pinned memory on host for faster transfers
        host_tensor = torch.randn(size_elements, dtype=torch.float32).pin_memory()
        device_tensor = torch.empty(size_elements, dtype=torch.float32, device=self.device)
        
        # Warmup
        for _ in range(10):
            device_tensor.copy_(host_tensor, non_blocking=False)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            device_tensor.copy_(host_tensor, non_blocking=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        bandwidth = (size_mb / 1024) / (time_ms / 1000)  # GB/s
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="Memory",
            operation=f"Host-to-Device (PCIe) {size_mb} MB",
            dtype="float32",
            shape=f"{size_elements}",
            throughput_tflops=0.0,
            time_ms=time_ms,
            memory_bandwidth_gb_s=bandwidth
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  Host-to-Device (H2D) {size_mb} MB: {bandwidth:.2f} GB/s, {time_ms:.3f} ms")
        
        del host_tensor, device_tensor
        torch.cuda.empty_cache()
        
        return bandwidth
    
    def benchmark_device_to_host(self, size_mb: int = 1024, iterations: int = 100):
        """Benchmark device-to-host (GPU to CPU) transfer bandwidth"""
        size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        
        device_tensor = torch.randn(size_elements, dtype=torch.float32, device=self.device)
        # Create pinned memory on host for faster transfers
        host_tensor = torch.empty(size_elements, dtype=torch.float32).pin_memory()
        
        # Warmup
        for _ in range(10):
            host_tensor.copy_(device_tensor, non_blocking=False)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            host_tensor.copy_(device_tensor, non_blocking=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        bandwidth = (size_mb / 1024) / (time_ms / 1000)  # GB/s
        
        result = BenchmarkResult(
            gpu_name=self.gpu_name,
            gpu_memory_gb=self.gpu_memory,
            cuda_capability=self.cuda_capability,
            test_name="Memory",
            operation=f"Device-to-Host (PCIe) {size_mb} MB",
            dtype="float32",
            shape=f"{size_elements}",
            throughput_tflops=0.0,
            time_ms=time_ms,
            memory_bandwidth_gb_s=bandwidth
        )
        self.results.append(result)
        
        if self.rank == 0:
            print(f"  Device-to-Host (D2H) {size_mb} MB: {bandwidth:.2f} GB/s, {time_ms:.3f} ms")
        
        del device_tensor, host_tensor
        torch.cuda.empty_cache()
        
        return bandwidth
    
    def run_full_suite(self, scale=1.0, matmul_size=None, batch_size=None, seq_length=None):
        """Run complete benchmark suite with optional scaling
        
        Args:
            scale: Multiplier for all dimensions (default: 1.0)
            matmul_size: Override matrix size for GEMM tests
            batch_size: Override batch size for conv/attention
            seq_length: Override sequence length for attention
        """
        self.warmup()
        
        # Calculate scaled sizes
        small_matmul = matmul_size if matmul_size else int(4096 * scale)
        large_matmul = matmul_size if matmul_size else int(8192 * scale)
        xlarge_matmul = matmul_size if matmul_size else int(16384 * scale)
        
        conv_batch = batch_size if batch_size else max(1, int(64 * scale))
        large_conv_batch = batch_size if batch_size else max(1, int(128 * scale))
        
        attn_batch_small = batch_size if batch_size else max(1, int(32 * scale))
        attn_batch_large = batch_size if batch_size else max(1, int(16 * scale))
        attn_batch_xlarge = batch_size if batch_size else max(1, int(8 * scale))
        
        seq_len_small = seq_length if seq_length else int(512 * scale)
        seq_len_large = seq_length if seq_length else int(2048 * scale)
        seq_len_xlarge = seq_length if seq_length else int(4096 * scale)
        
        mem_size_small = int(512 * scale)
        mem_size_medium = int(1024 * scale)
        mem_size_large = int(2048 * scale)
        
        if self.rank == 0:
            print(f"\nScale Factor: {scale}x")
            if matmul_size:
                print(f"Custom MatMul Size: {matmul_size}")
            if batch_size:
                print(f"Custom Batch Size: {batch_size}")
            if seq_length:
                print(f"Custom Sequence Length: {seq_length}")
        
        if self.rank == 0:
            print("=" * 70)
            print("MATRIX MULTIPLICATION (GEMM) BENCHMARKS")
            print("=" * 70)
        
        # Small GEMM
        if self.rank == 0:
            print(f"\nSmall GEMM ({small_matmul}x{small_matmul}x{small_matmul}):")
        self.benchmark_matmul(small_matmul, small_matmul, small_matmul, torch.float32)
        self.benchmark_matmul(small_matmul, small_matmul, small_matmul, torch.float16)
        if hasattr(torch, 'bfloat16'):
            self.benchmark_matmul(small_matmul, small_matmul, small_matmul, torch.bfloat16)
        
        # Large GEMM
        if self.rank == 0:
            print(f"\nLarge GEMM ({large_matmul}x{large_matmul}x{large_matmul}):")
        self.benchmark_matmul(large_matmul, large_matmul, large_matmul, torch.float32)
        self.benchmark_matmul(large_matmul, large_matmul, large_matmul, torch.float16)
        if hasattr(torch, 'bfloat16'):
            self.benchmark_matmul(large_matmul, large_matmul, large_matmul, torch.bfloat16)
        
        # Very Large GEMM (tensor core optimized)
        if self.rank == 0:
            print(f"\nVery Large GEMM ({xlarge_matmul}x{xlarge_matmul}x{xlarge_matmul}):")
        self.benchmark_matmul(xlarge_matmul, xlarge_matmul, xlarge_matmul, torch.float16, iterations=20)
        if hasattr(torch, 'bfloat16'):
            self.benchmark_matmul(xlarge_matmul, xlarge_matmul, xlarge_matmul, torch.bfloat16, iterations=20)
        
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("CONVOLUTION BENCHMARKS")
            print("=" * 70)
        
        # ResNet-like convolutions
        if self.rank == 0:
            print("\nResNet-style Conv2D:")
        self.benchmark_conv2d(conv_batch, 64, 64, 224, 224, 3, torch.float32)
        self.benchmark_conv2d(conv_batch, 64, 64, 224, 224, 3, torch.float16)
        self.benchmark_conv2d(large_conv_batch, 128, 128, 112, 112, 3, torch.float16)
        self.benchmark_conv2d(max(1, int(256 * scale)), 256, 256, 56, 56, 3, torch.float16)
        
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("ATTENTION BENCHMARKS (Transformer)")
            print("=" * 70)
        
        # Various attention configurations
        if self.rank == 0:
            print("\nSmall Attention (BERT-like):")
        self.benchmark_attention(attn_batch_small, seq_len_small, 12, 64, torch.float32)
        self.benchmark_attention(attn_batch_small, seq_len_small, 12, 64, torch.float16)
        
        if self.rank == 0:
            print("\nLarge Attention (GPT-like):")
        self.benchmark_attention(attn_batch_large, seq_len_large, 32, 128, torch.float16)
        if hasattr(torch, 'bfloat16'):
            self.benchmark_attention(attn_batch_large, seq_len_large, 32, 128, torch.bfloat16)
        
        if self.rank == 0:
            print("\nVery Large Attention:")
        self.benchmark_attention(attn_batch_xlarge, seq_len_xlarge, 40, 128, torch.float16, iterations=20)
        
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("MEMORY BANDWIDTH BENCHMARKS")
            print("=" * 70)
        
        if self.rank == 0:
            print("\nDevice-to-Device (GPU Memory):")
        self.benchmark_memory_bandwidth(mem_size_small)
        self.benchmark_memory_bandwidth(mem_size_medium)
        self.benchmark_memory_bandwidth(mem_size_large)
        
        if self.rank == 0:
            print("\nHost-to-Device (CPU -> GPU via PCIe/SXM):")
        self.benchmark_host_to_device(mem_size_small)
        self.benchmark_host_to_device(mem_size_medium)
        self.benchmark_host_to_device(mem_size_large)
        
        if self.rank == 0:
            print("\nDevice-to-Host (GPU -> CPU via PCIe/SXM):")
        self.benchmark_device_to_host(mem_size_small)
        self.benchmark_device_to_host(mem_size_medium)
        self.benchmark_device_to_host(mem_size_large)
        
        if self.rank == 0:
            print("\n" + "=" * 70)
            print("BENCHMARK COMPLETE")
            print("=" * 70)
        
        # Multi-GPU specific tests
        if self.multi_gpu and self.world_size > 1:
            if self.rank == 0:
                print("\n" + "=" * 70)
                print("MULTI-GPU BENCHMARKS")
                print("=" * 70)
            
            if self.rank == 0:
                print("\nData Parallel Training:")
            self.benchmark_data_parallel(1024, 1024, 1024, torch.float16)
            self.benchmark_data_parallel(4096, 4096, 4096, torch.float16)
            
            if self.rank == 0:
                print("\nCollective Communication (All-Reduce):")
            self.benchmark_all_reduce(10)
            self.benchmark_all_reduce(100)
            self.benchmark_all_reduce(500)
            
            if self.rank == 0:
                print("\nPeer-to-Peer Transfer:")
            self.benchmark_peer_to_peer_nccl(10)
            self.benchmark_peer_to_peer_nccl(100)
            self.benchmark_peer_to_peer_nccl(500)
            
            if self.rank == 0:
                print("\n" + "=" * 70)
                print("MULTI-GPU BENCHMARK COMPLETE")
                print("=" * 70)
    
    def print_summary(self):
        """Print summary statistics"""
        if self.rank != 0:
            return
            
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        # Group by test type
        test_types = {}
        for result in self.results:
            if result.test_name not in test_types:
                test_types[result.test_name] = []
            test_types[result.test_name].append(result)
        
        for test_name, results in test_types.items():
            print(f"\n{test_name}:")
            
            # Check if this is a bandwidth-only test (no FLOPS)
            has_flops = any(r.throughput_tflops > 0 for r in results)
            has_bandwidth = any(r.memory_bandwidth_gb_s > 0 for r in results)
            
            if has_bandwidth and not has_flops:
                # Bandwidth-only tests (Memory, AllReduce, P2P)
                max_bw = max(r.memory_bandwidth_gb_s for r in results)
                avg_bw = sum(r.memory_bandwidth_gb_s for r in results) / len(results)
                print(f"  Max Bandwidth: {max_bw:.2f} GB/s")
                print(f"  Avg Bandwidth: {avg_bw:.2f} GB/s")
                
            elif has_flops:
                # Compute tests (GEMM, Conv2D, Attention, DataParallel)
                max_tflops = max(r.throughput_tflops for r in results)
                avg_tflops = sum(r.throughput_tflops for r in results) / len(results)
                
                # Filter by dtype
                fp32_results = [r for r in results if 'float32' in r.dtype]
                fp16_results = [r for r in results if 'float16' in r.dtype]
                bf16_results = [r for r in results if 'bfloat16' in r.dtype]
                
                print(f"  Max: {max_tflops:.2f} TFLOPS")
                print(f"  Avg: {avg_tflops:.2f} TFLOPS")
                
                if fp32_results:
                    fp32_max = max(r.throughput_tflops for r in fp32_results)
                    print(f"  FP32 Max: {fp32_max:.2f} TFLOPS")
                
                if fp16_results:
                    fp16_max = max(r.throughput_tflops for r in fp16_results)
                    print(f"  FP16 Max: {fp16_max:.2f} TFLOPS")
                
                if bf16_results:
                    bf16_max = max(r.throughput_tflops for r in bf16_results)
                    print(f"  BF16 Max: {bf16_max:.2f} TFLOPS")
    
    def benchmark_data_parallel(self, M: int, N: int, K: int, dtype: torch.dtype, iterations: int = 50):
        """Benchmark data parallel training across multiple GPUs"""
        if not self.multi_gpu:
            if self.rank == 0:
                print("  Skipping: Multi-GPU mode not enabled")
            return 0
        
        # Create model and explicitly specify device_ids for DDP
        model = nn.Linear(K, N, dtype=dtype).to(self.device)
        # Specify device_ids to tell DDP exactly which device this rank uses
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        
        # Each GPU gets a portion of the batch
        local_batch = M // self.world_size
        x = torch.randn(local_batch, K, dtype=dtype, device=self.device)
        target = torch.randn(local_batch, N, dtype=dtype, device=self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if self.multi_gpu:
            dist.barrier()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        if self.multi_gpu:
            dist.barrier()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        
        # Calculate FLOPS (forward + backward ~3x forward)
        flops_per_iter = 2 * M * N * K * 3  # Approximate
        tflops = (flops_per_iter / (time_ms / 1000)) / 1e12
        
        if self.rank == 0:
            result = BenchmarkResult(
                gpu_name=self.gpu_name,
                gpu_memory_gb=self.gpu_memory,
                cuda_capability=self.cuda_capability,
                test_name="DataParallel",
                operation=f"DDP Training {M}x{K}x{N} on {self.world_size} GPUs",
                dtype=str(dtype).split('.')[-1],
                shape=f"{M}x{K}x{N}",
                throughput_tflops=tflops,
                time_ms=time_ms
            )
            self.results.append(result)
            print(f"  Data Parallel {M}x{K}x{N} ({dtype}) on {self.world_size} GPUs: {tflops:.2f} TFLOPS, {time_ms:.3f} ms")
        
        del model, x, target, optimizer
        torch.cuda.empty_cache()
        
        return tflops
    
    def benchmark_all_reduce(self, size_mb: int = 100, iterations: int = 100):
        """Benchmark all-reduce collective operation (gradient synchronization)"""
        if not self.multi_gpu:
            if self.rank == 0:
                print("  Skipping: Multi-GPU mode not enabled")
            return 0
        
        size_elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        tensor = torch.randn(size_elements, dtype=torch.float32, device=self.device)
        
        # Warmup
        for _ in range(10):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        dist.barrier()
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        dist.barrier()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        time_ms = (end - start) * 1000 / iterations
        
        # All-reduce transfers: (N-1)/N * 2 * size where N is world_size
        # Ring all-reduce algorithm
        bandwidth = (size_mb / 1024) * 2 * (self.world_size - 1) / self.world_size / (time_ms / 1000)  # GB/s
        
        if self.rank == 0:
            result = BenchmarkResult(
                gpu_name=self.gpu_name,
                gpu_memory_gb=self.gpu_memory,
                cuda_capability=self.cuda_capability,
                test_name="AllReduce",
                operation=f"All-Reduce {size_mb} MB on {self.world_size} GPUs",
                dtype="float32",
                shape=f"{size_elements}",
                throughput_tflops=0.0,
                time_ms=time_ms,
                memory_bandwidth_gb_s=bandwidth
            )
            self.results.append(result)
            print(f"  All-Reduce {size_mb} MB on {self.world_size} GPUs: {bandwidth:.2f} GB/s, {time_ms:.3f} ms")
        
        del tensor
        torch.cuda.empty_cache()
        
        return bandwidth
    
    
    def benchmark_peer_to_peer_nccl(self, size_mb=100, iterations=100):
        """
        NCCL-tests style P2P benchmark (closest PyTorch equivalent)

        Key idea:
        - ONE NCCL call per iteration
        - NO Python-level multi-op scheduling
        - Persistent buffers
        - Timing only around full collective execution
        """

        if not self.multi_gpu or self.world_size < 2:
            if self.rank == 0:
                print("Requires at least 2 GPUs")
            return 0

        size_elements = (size_mb * 1024 * 1024) // 4  # float32

        # Persistent buffers (IMPORTANT: no reallocation)
        send = torch.randn(size_elements, dtype=torch.float32, device=self.device)
        recv = torch.empty(size_elements, dtype=torch.float32, device=self.device)

        # Warmup (NCCL autotuning phase)
        for _ in range(10):
            dist.all_reduce(send, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        dist.barrier()

        # Timing
        start = time.perf_counter()

        for _ in range(iterations):
            # IMPORTANT: single NCCL call per iteration
            # This is what makes it similar to nccl-tests
            dist.all_reduce(send, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize()
        dist.barrier()

        end = time.perf_counter()

        # Compute time
        time_ms = (end - start) * 1000 / iterations

        # Bandwidth calculation (nccl-tests style)
        # Each GPU communicates with (world_size - 1) peers implicitly
        total_bytes = size_mb * 1024 * 1024
        # total_traffic = 2 * total_bytes * (self.world_size - 1)
        total_traffic = total_bytes * (self.world_size - 1)
        bandwidth = (total_traffic / (time_ms / 1000)) / 1e9

        if self.rank == 0:
            print(f"[NCCL-style P2P] {size_mb}MB: {bandwidth:.2f} GB/s, {time_ms:.3f} ms")

        return bandwidth if self.rank == 0 else 0

    def benchmark_peer_to_peer_streamed(self, size_mb: int = 512, iterations: int = 50):
        """
        High-performance P2P benchmark (NVLink/PCIe)
        - All ranks participate
        - Multi-peer communication
        - Multiple concurrent transfers
        - CUDA streams for overlap
        """
        if not self.multi_gpu or self.world_size < 2:
            if self.rank == 0:
                print("  Skipping: Requires at least 2 GPUs")
            return 0

        size_elements = (size_mb * 1024 * 1024) // 4  # float32
        num_chunks = 4  # increase to 8 for more saturation on big GPUs

        # Allocate multiple buffers for concurrency
        send_tensors = [
            torch.randn(size_elements, dtype=torch.float32, device=self.device)
            for _ in range(num_chunks)
        ]
        recv_tensors = [
            torch.empty(size_elements, dtype=torch.float32, device=self.device)
            for _ in range(num_chunks)
        ]

        # Create CUDA streams for overlap
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_chunks)]

        def run_p2p():
            reqs = []

            for chunk in range(num_chunks):
                with torch.cuda.stream(streams[chunk]):
                    ops = []
                    for peer in range(self.world_size):
                        if peer == self.rank:
                            continue

                        ops.append(dist.P2POp(dist.isend, send_tensors[chunk], peer))
                        ops.append(dist.P2POp(dist.irecv, recv_tensors[chunk], peer))

                    reqs.extend(dist.batch_isend_irecv(ops))

            return reqs

        # Warmup
        for _ in range(10):
            reqs = run_p2p()
            for r in reqs:
                r.wait()

        dist.barrier()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()

        for _ in range(iterations):
            reqs = run_p2p()

        # Wait AFTER loop (important for overlap)
        for r in reqs:
            r.wait()

        torch.cuda.synchronize()
        dist.barrier()

        end = time.perf_counter()

        time_ms = (end - start) * 1000 / iterations

        bytes = size_mb * 1024 * 1024
        time_s = time_ms / 1000

        bandwidth = bytes / time_s / 1e9  # injected BW (GB/s)

        if self.rank == 0:
            result = BenchmarkResult(
                gpu_name=self.gpu_name,
                gpu_memory_gb=self.gpu_memory,
                cuda_capability=self.cuda_capability,
                test_name="P2P",
                operation=f"Full Mesh P2P {size_mb}MB x{num_chunks} on {self.world_size} GPUs",
                dtype="float32",
                shape=f"{size_elements}",
                throughput_tflops=0.0,
                time_ms=time_ms,
                memory_bandwidth_gb_s=bandwidth
            )
            self.results.append(result)

            print(f"  P2P Full Mesh ({self.world_size} GPUs, {size_mb} MB x{num_chunks}): "
                f"{bandwidth:.2f} GB/s, {time_ms:.3f} ms")

        # Cleanup
        del send_tensors, recv_tensors, streams
        torch.cuda.empty_cache()

        return bandwidth if self.rank == 0 else 0

    def benchmark_peer_to_peer_orig(self, size_mb: int = 512, iterations: int = 100):
        """Benchmark P2P GPU-to-GPU transfer using batch_isend_irecv for true simultaneous bidirectional NVLink transfer"""
        if not self.multi_gpu or self.world_size < 2:
            if self.rank == 0:
                print("  Skipping: Requires at least 2 GPUs")
            return 0

        size_elements = (size_mb * 1024 * 1024) // 4
        bandwidth = 0

        
        if self.rank in (0, 1):
            peer = 1 - self.rank  # rank 0 <-> rank 1
            send_tensor = torch.randn(size_elements, dtype=torch.float32, device=self.device)
            recv_tensor = torch.empty(size_elements, dtype=torch.float32, device=self.device)

            def do_p2p():
                ops = [
                    dist.P2POp(dist.isend, send_tensor, peer),
                    dist.P2POp(dist.irecv, recv_tensor, peer),
                ]
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
        
            """
            def do_p2p_all():
                ops = []
                for peer in range(self.world_size):
                    if peer != self.rank:
                        ops.append(dist.P2POp(dist.isend, send_tensor, peer))
                        ops.append(dist.P2POp(dist.irecv, recv_tensor, peer))

                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()        
            """ 
            # Warmup
            for _ in range(10):
                do_p2p()
            dist.barrier()
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(iterations):
                do_p2p()
            dist.barrier()
            torch.cuda.synchronize()
            end = time.perf_counter()

            time_ms = (end - start) * 1000 / iterations
            
            bytes = size_mb * 1024 * 1024
            time_s = time_ms / 1000

            #bw_one_dir = bytes / time_s / 1e9
            bandwidth  = 2 * bytes / time_s / 1e9

            #bandwidth = (size_mb / 1024) * 2 / (time_ms / 1000)  # GB/s bidirectional

            if self.rank == 0:
                result = BenchmarkResult(
                    gpu_name=self.gpu_name,
                    gpu_memory_gb=self.gpu_memory,
                    cuda_capability=self.cuda_capability,
                    test_name="P2P",
                    operation=f"P2P Transfer {size_mb} MB (GPU 0 <-> GPU 1)",
                    dtype="float32",
                    shape=f"{size_elements}",
                    throughput_tflops=0.0,
                    time_ms=time_ms,
                    memory_bandwidth_gb_s=bandwidth
                )
                self.results.append(result)
                print(f"  P2P Transfer {size_mb} MB (bidirectional): {bandwidth:.2f} GB/s, {time_ms:.3f} ms")

            del send_tensor, recv_tensor
        else:
            # Ranks 2+ sit out both the warmup barrier and benchmark barrier
            dist.barrier()
            dist.barrier()

        torch.cuda.empty_cache()
        return bandwidth if self.rank == 0 else 0
    

    def save_results_json(self, filename: str = "gpu_benchmark_results.json"):
        """Save results to JSON file"""
        if self.rank != 0:
            return
            
        data = {
            'gpu_info': {
                'name': self.gpu_name,
                'memory_gb': self.gpu_memory,
                'cuda_capability': self.cuda_capability,
                'multi_gpu': self.multi_gpu,
                'world_size': self.world_size
            },
            'results': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_results(self, filename: str = "gpu_benchmark_results.yaml"):
        """Save results to YAML file"""
        if self.rank != 0:
            return
            
        data = {
            'gpu_info': {
                'name': self.gpu_name,
                'memory_gb': float(self.gpu_memory),
                'cuda_capability': self.cuda_capability,
                'multi_gpu': self.multi_gpu,
                'world_size': self.world_size
            },
            'results': [asdict(r) for r in self.results]
        }
        
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nResults saved to {filename}")

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    # Set environment variables before init_process_group
    os.environ['MASTER_ADDR'] = 'localhost'
    # MASTER_PORT is set by main() before spawning; fall back to 12355
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    
    # Set the CUDA device for this process BEFORE init_process_group
    # This ensures NCCL knows which GPU this rank is using
    torch.cuda.set_device(rank)
    
    # Initialize process group
    # Note: NCCL will automatically use the device set by torch.cuda.set_device()
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Verify the device is set correctly
    current_device = torch.cuda.current_device()
    if current_device != rank:
        raise RuntimeError(
            f"Rank {rank} expected device {rank} but got device {current_device}"
        )

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def run_benchmark_distributed(rank, world_size, args):
    """Run benchmark on a single GPU in distributed mode"""
    setup_distributed(rank, world_size)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("NVIDIA GPU ML BENCHMARK SUITE - MULTI-GPU MODE")
        print(f"Running on {world_size} GPUs")
        print("Architectures: A100, H100, L40S, B200/B300")
        print("=" * 70)
        print("\nNote: You may see NCCL warnings about 'device used by this process")
        print("      is currently unknown' - these are harmless and can be ignored.")
        print("=" * 70)
    
    benchmark = GPUBenchmark(
        device_id=rank, 
        multi_gpu=True, 
        rank=rank, 
        world_size=world_size
    )
    
    benchmark.run_full_suite(
        scale=args.scale,
        matmul_size=args.matmul_size,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
    
    benchmark.print_summary()
    benchmark.save_results(args.output)
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("Multi-GPU Benchmark completed successfully!")
        print("=" * 70)
    
    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description='NVIDIA GPU ML Benchmark Suite')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID (default: 0)')
    parser.add_argument('--output', type=str, default='gpu_benchmark_results.yaml', 
                       help='Output YAML file (default: gpu_benchmark_results.yaml)')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick benchmark (fewer iterations)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for data sizes (default: 1.0, larger = bigger tensors)')
    parser.add_argument('--matmul-size', type=int, default=None,
                       help='Custom matrix size for GEMM (overrides scale)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Custom batch size for conv/attention (overrides scale)')
    parser.add_argument('--seq-length', type=int, default=None,
                       help='Custom sequence length for attention (overrides scale)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable multi-GPU benchmarking (uses all available GPUs)')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--preset', type=str, default=None,
                       choices=['small', 'medium', 'large', 'xlarge'],
                       help='Preset configuration: small (8-16GB), medium (24-40GB), large (40-80GB), xlarge (80GB+)')
    
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset:
        presets = {
            'small': {
                'scale': 0.35,
                'description': 'Optimized for 8-16GB GPUs (GTX 1080 Ti, RTX 3060, Quadro GP100, etc.)'
            },
            'medium': {
                'scale': 0.75,
                'description': 'Optimized for 24-40GB GPUs (RTX 3090, RTX 4090, A10, A100 40GB, etc.)'
            },
            'large': {
                'scale': 1.5,
                'description': 'Optimized for 40-80GB GPUs (A100 80GB, H100, L40S, etc.)'
            },
            'xlarge': {
                'scale': 2.5,
                'description': 'Optimized for 80GB+ GPUs (H100, B200, etc.)'
            }
        }
        
        preset_config = presets[args.preset]
        args.scale = preset_config['scale']
        print(f"\nUsing preset: {args.preset}")
        print(f"Description: {preset_config['description']}")
        print(f"Scale factor: {args.scale}x\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a NVIDIA GPU.")
        sys.exit(1)
    
    # Multi-GPU mode
    if args.multi_gpu:
        world_size = args.num_gpus if args.num_gpus else torch.cuda.device_count()
        
        if world_size < 2:
            print("ERROR: Multi-GPU mode requires at least 2 GPUs.")
            print(f"Found {world_size} GPU(s).")
            sys.exit(1)
        
        # Pick a free port and export it so all spawned ranks inherit it
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
            _s.bind(('', 0))
            os.environ['MASTER_PORT'] = str(_s.getsockname()[1])

        print(f"\nLaunching multi-GPU benchmark on {world_size} GPUs...")
        mp.spawn(
            run_benchmark_distributed,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU mode
        print("\n" + "=" * 70)
        print("NVIDIA GPU ML BENCHMARK SUITE")
        print("Architectures: A100, H100, L40S, B200/B300")
        print("=" * 70)
        
        benchmark = GPUBenchmark(device_id=args.device)
        benchmark.run_full_suite(
            scale=args.scale,
            matmul_size=args.matmul_size,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        benchmark.print_summary()
        benchmark.save_results(args.output)
        
        print("\n" + "=" * 70)
        print("Benchmark completed successfully!")
        print("=" * 70)

if __name__ == "__main__":
    main()
