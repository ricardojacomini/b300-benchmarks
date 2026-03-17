#!/usr/bin/env python3
"""
AlphaFold2 Evoformer Proxy Benchmark — NVIDIA B300 SXM6
Measures JAX JIT attention throughput as a proxy for AF2 inference speed.
No model weights required.
"""
import os, time, json
import jax
import jax.numpy as jnp
import numpy as np

print(f"JAX version:  {jax.__version__}", flush=True)
print(f"JAX devices:  {jax.devices()}", flush=True)
print(f"GPU:          {jax.devices()[0].device_kind}", flush=True)
print(f"XLA backend:  {jax.default_backend()}", flush=True)

# Evoformer-equivalent attention proxy (single-sequence mode)
def evoformer_proxy(seq_len, n_warmup=3, n_runs=10):
    d_model = 256
    n_heads = 8
    key = jax.random.PRNGKey(42)

    @jax.jit
    def attn(q, k, v):
        scale = (d_model // n_heads) ** -0.5
        logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        weights = jax.nn.softmax(logits, axis=-1)
        return jnp.einsum("bhqk,bhkd->bhqd", weights, v)

    shape = (1, n_heads, seq_len, d_model // n_heads)
    q = jax.random.normal(key, shape, dtype=jnp.float16)
    k = jax.random.normal(key, shape, dtype=jnp.float16)
    v = jax.random.normal(key, shape, dtype=jnp.float16)

    for _ in range(n_warmup):
        _ = attn(q, k, v).block_until_ready()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = attn(q, k, v).block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.min(times))

sep = "=" * 62
print(f"\n{sep}", flush=True)
print("  Evoformer Attention Proxy Benchmark (JAX JIT, FP16)", flush=True)
print(sep, flush=True)
print(f"  {'seq_len':>8}  {'mean (ms)':>10}  {'min (ms)':>10}", flush=True)
print("-" * 40, flush=True)

results = {}
for seq_len in [256, 384, 512, 769]:
    mean_ms, min_ms = evoformer_proxy(seq_len)
    results[seq_len] = {"mean_ms": round(mean_ms, 2), "min_ms": round(min_ms, 2)}
    print(f"  {seq_len:>8}  {mean_ms:>10.2f}  {min_ms:>10.2f}", flush=True)

print(sep, flush=True)
print("  Reference full AF2 inference (T1049, 769 res, single model):", flush=True)
print("    H100 SXM5:  ~11.5 min", flush=True)
print("    B200 SXM:   ~8.5 min (estimated)", flush=True)
print("    B300 SXM6:  TBD — mount weights + run run_alphafold.py", flush=True)
print(sep, flush=True)

os.makedirs("/workspace/logs", exist_ok=True)
out = {
    "gpu": str(jax.devices()[0]),
    "jax_version": jax.__version__,
    "seq_results": results,
}
with open("/workspace/logs/alphafold_b300_proxy.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nProxy results saved to /workspace/logs/alphafold_b300_proxy.json", flush=True)
