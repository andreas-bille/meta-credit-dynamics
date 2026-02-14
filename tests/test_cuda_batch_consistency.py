from __future__ import annotations

import os
import random

import numpy as np
import torch

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld
from capitalmarket.capitalselector.cpu_impl import CpuCore
from capitalmarket.capitalselector.cuda_impl import CudaCore


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _collect_world_outputs(steps: int, seed: int):
    world = RegimeSwitchBanditWorld(p=0.05, sigma=0.01, seed=seed)
    outputs = []
    for t in range(steps):
        out = world.step(t)
        r_vec, c_total = validate_world_output(out)
        outputs.append((np.asarray(r_vec, dtype=float), float(c_total)))
    return outputs


def _init_selector(K: int):
    selector = CapitalSelectorBuilder().with_K(K).build()
    selector.w = np.ones(K) / max(1, K)
    selector.K = K
    return selector


def _assert_final_parity(seed: int, selector_cpu, selector_cuda) -> None:
    np.testing.assert_allclose(selector_cuda.w.sum(), 1.0, rtol=0.0, atol=1e-12)
    assert not np.isnan(selector_cuda.w).any()

    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.wealth, selector_cpu.wealth, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.stats.mu, selector_cpu.stats.mu, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.stats.var, selector_cpu.stats.var, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.stats.cum_pi, selector_cpu.stats.cum_pi, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.stats.peak_cum_pi, selector_cpu.stats.peak_cum_pi, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.stats.dd, selector_cpu.stats.dd, rtol=0.0, atol=1e-15)


def test_cuda_batch_consistency_b2():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"

    steps = 500
    seed0 = 0
    seed1 = 1

    _seed_all(seed0)
    outputs0 = _collect_world_outputs(steps=steps, seed=seed0)
    _seed_all(seed1)
    outputs1 = _collect_world_outputs(steps=steps, seed=seed1)

    r_vec0, _ = outputs0[0]
    r_vec1, _ = outputs1[0]
    K = len(r_vec0)
    assert len(r_vec1) == K

    selector_cpu0 = _init_selector(K)
    selector_cpu1 = _init_selector(K)
    selector_cuda0 = _init_selector(K)
    selector_cuda1 = _init_selector(K)

    cpu_core = CpuCore()
    cuda_core = CudaCore()

    for (r0, c0), (r1, c1) in zip(outputs0, outputs1):
        cpu_core.step(selector_cpu0, r0, c0, freeze=False)
        cpu_core.step(selector_cpu1, r1, c1, freeze=False)

        # Batch-lift is not yet available; emulate B=2 by lockstep stepping.
        cuda_core.step(selector_cuda0, r0, c0, freeze=False)
        cuda_core.step(selector_cuda1, r1, c1, freeze=False)

    _assert_final_parity(seed0, selector_cpu0, selector_cuda0)
    _assert_final_parity(seed1, selector_cpu1, selector_cuda1)
