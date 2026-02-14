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


def _collect_world_outputs(steps: int, seed: int = 0):
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


def test_cuda_rebirth_forced_parity():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    _seed_all(0)

    outputs = _collect_world_outputs(steps=1, seed=0)
    r_vec, c_total = outputs[0]
    K = len(r_vec)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    selector_cpu.wealth = 0.0
    selector_cpu.rebirth_threshold = 1.0
    selector_cuda.wealth = 0.0
    selector_cuda.rebirth_threshold = 1.0

    cpu_core = CpuCore()
    cuda_core = CudaCore()

    cpu_core.step(selector_cpu, r_vec, c_total, freeze=False)
    cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)

    # Minimal tolerance to account for tiny float64 accumulation differences.
    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(selector_cuda.wealth, selector_cpu.wealth, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selector_cuda.stats.mu, selector_cpu.stats.mu, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selector_cuda.stats.var, selector_cpu.stats.var, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selector_cuda.stats.dd, selector_cpu.stats.dd, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selector_cuda.stats.cum_pi, selector_cpu.stats.cum_pi, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(selector_cuda.stats.peak_cum_pi, selector_cpu.stats.peak_cum_pi, rtol=0.0, atol=0.0)


def test_cuda_rebirth_no_reset():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    _seed_all(0)

    outputs = _collect_world_outputs(steps=1, seed=0)
    r_vec, c_total = outputs[0]
    K = len(r_vec)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    cpu_core = CpuCore()
    cuda_core = CudaCore()

    cpu_core.step(selector_cpu, r_vec, c_total, freeze=False)
    cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)

    # Minimal tolerance to account for tiny float64 accumulation differences.
    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)


def test_cuda_rebirth_multi_step_stability():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    _seed_all(0)

    outputs = _collect_world_outputs(steps=50, seed=0)
    r_vec0, _ = outputs[0]
    K = len(r_vec0)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    cpu_core = CpuCore()
    cuda_core = CudaCore()

    for r_vec, c_total in outputs:
        cpu_core.step(selector_cpu, r_vec, c_total, freeze=False)
        cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)

    # Minimal tolerance to account for tiny float64 accumulation differences.
    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)
