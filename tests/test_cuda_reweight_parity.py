from __future__ import annotations

import os
import random

import numpy as np
import torch

from capitalmarket.capitalselector.builder import CapitalSelectorBuilder
from capitalmarket.capitalselector.interfaces import validate_world_output
from capitalmarket.capitalselector.worlds.regime_switch_bandit_world import RegimeSwitchBanditWorld
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


def test_cuda_reweight_single_step_parity():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    _seed_all(0)

    outputs = _collect_world_outputs(steps=1, seed=0)
    r_vec, c_total = outputs[0]
    K = len(r_vec)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    cuda_core = CudaCore()

    # CPU reference reweight
    pi_vec = r_vec - selector_cpu.w * c_total
    score = pi_vec - float(selector_cpu.stats.mu)
    selector_cpu.w = selector_cpu.reweight_fn(selector_cpu.w, score)

    # CUDA reweight
    cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)

    # Minimal tolerance to account for tiny float64 accumulation differences.
    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)


def test_cuda_reweight_multi_step_drift():
    os.environ["CAPM_ENABLE_TOPOLOGY"] = "0"
    _seed_all(0)

    outputs = _collect_world_outputs(steps=20, seed=0)
    r_vec0, _ = outputs[0]
    K = len(r_vec0)

    selector_cpu = _init_selector(K)
    selector_cuda = _init_selector(K)

    cuda_core = CudaCore()

    for r_vec, c_total in outputs:
        pi_vec = r_vec - selector_cpu.w * c_total
        score = pi_vec - float(selector_cpu.stats.mu)
        selector_cpu.w = selector_cpu.reweight_fn(selector_cpu.w, score)

        cuda_core.step(selector_cuda, r_vec, c_total, freeze=False)

    # Minimal tolerance to account for tiny float64 accumulation differences.
    np.testing.assert_allclose(selector_cuda.w, selector_cpu.w, rtol=0.0, atol=1e-15)
