from __future__ import annotations

import os
import random
import time

import numpy as np
import pytest
import torch

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.cpu_impl import CpuCore
from capitalmarket.capitalselector.cuda_impl import CudaCore


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _collect_synthetic_outputs(steps: int, K: int, seed: int, p: float = 0.05, sigma: float = 0.01):
    rng = np.random.default_rng(int(seed))
    regime = "A" if rng.random() < 0.5 else "B"
    outputs: list[tuple[np.ndarray, float]] = []
    for t in range(steps):
        _ = t
        if rng.random() < float(p):
            regime = "B" if regime == "A" else "A"
        mean = np.zeros(K, dtype=float)
        if regime == "A":
            mean[0] = 0.02
        else:
            mean[3] = 0.02
        noise = rng.normal(0.0, float(sigma), size=mean.shape)
        r_vec = mean + noise
        outputs.append((r_vec, 0.0))
    return outputs


def _init_selector(K: int):
    selector = CapitalSelectorBuilder().with_K(K).build()
    selector.w = np.ones(K) / max(1, K)
    selector.K = K
    return selector


@pytest.mark.skipif(os.environ.get("CAPM_BENCHMARK") != "1", reason="Set CAPM_BENCHMARK=1 to run")
def test_cuda_benchmark_optional():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"

    seed = 0
    steps = 200
    K = 10_000

    _seed_all(seed)
    outputs = _collect_synthetic_outputs(steps=steps, K=K, seed=seed)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    cpu_core = CpuCore()
    cuda_core = CudaCore()

    t0 = time.perf_counter()
    for r_vec, c_total in outputs:
        cpu_core.step(selector_cpu, r_vec, c_total, freeze=False)
    cpu_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    for r_vec, c_total in outputs:
        cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)
    cuda_time = time.perf_counter() - t1

    print(f"CPU runtime: {cpu_time:.4f}s")
    print(f"CUDA runtime: {cuda_time:.4f}s")
    if torch.cuda.is_available():
        print(f"Speedup: {cpu_time / cuda_time:.2f}x")
    else:
        print(f"Speedup (CPU-vs-CPU, CudaCore on CPU): {cpu_time / cuda_time:.2f}x")

    assert not np.isnan(selector_cuda.w).any()
    np.testing.assert_allclose(selector_cuda.w.sum(), 1.0, rtol=0.0, atol=1e-12)
