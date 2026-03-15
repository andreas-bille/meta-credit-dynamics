from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict
import os
import numpy as np
import torch

from .config import ProfileAConfig
from .interfaces import World, Curriculum, Teacher, validate_world_output
from .builder import CapitalSelectorBuilder
from .cpu_impl import CpuCore
from .cuda_impl import CudaCore
from .determinism import enable_determinism
from .phase_i_state import DEFAULT_LAMBDA_RISK
from .population_manager import PopulationManager, RebirthConfig
from .selector_policy import build_world_action
from .selector_policy import DEFAULT_SELECTOR_POLICY, SelectorPolicy


@dataclass(frozen=True)
class RuntimeConfig:
    profile: str = "A"
    freeze: bool = False
    mode: str = "A"
    deterministic: bool = False
    seed: int | None = 0
    backend: str | None = None
    capm_mode: str | None = None
    config_backend: str | None = None
    config_mode: str | None = None
    max_claims_per_process: int = 1_000_000
    enable_meta_rebirth: bool = False
    rebirth_base_liquidity: float = 0.0
    rebirth_eta: float = 0.0
    rebirth_kappa: float = 1.0
    selector_policy: SelectorPolicy = DEFAULT_SELECTOR_POLICY
    lambda_risk: float = DEFAULT_LAMBDA_RISK


def _resolve_backend(cfg: RuntimeConfig) -> tuple[str, str]:
    env_backend = os.environ.get("CAPM_BACKEND", os.environ.get("CAPM_DEVICE", None))
    requested = cfg.backend if cfg.backend is not None else (env_backend if env_backend is not None else (cfg.config_backend if cfg.config_backend is not None else "cpu"))
    requested_norm = str(requested).strip().lower()
    if requested_norm == "gpu":
        requested_norm = "cuda"
    if requested_norm not in {"cpu", "cuda"}:
        raise RuntimeError(f"invalid backend '{requested}'; expected 'cpu' or 'cuda'")

    if requested_norm == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("backend=cuda requested but torch.cuda.is_available() is False")

    return requested_norm, requested_norm


def _resolve_capm_mode(cfg: RuntimeConfig) -> str:
    env_mode = os.environ.get("CAPM_MODE", None)
    mode = cfg.capm_mode if cfg.capm_mode is not None else (env_mode if env_mode is not None else (cfg.config_mode if cfg.config_mode is not None else "deterministic"))
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"deterministic", "fast"}:
        raise RuntimeError(f"invalid CAPM_MODE '{mode}'; expected 'deterministic' or 'fast'")
    return mode_norm


def _validate_builder_runtime(cfg: RuntimeConfig, *, effective_backend: str, effective_mode: str) -> None:
    if int(cfg.max_claims_per_process) <= 0:
        raise RuntimeError("max_claims_per_process must be > 0")

    dtype_env = os.environ.get("CAPM_DTYPE", "").strip().lower()
    if dtype_env and dtype_env not in {"float32", "float64"}:
        raise RuntimeError("CAPM_DTYPE must be 'float32' or 'float64' when set")

    if effective_mode == "deterministic" and cfg.seed is None:
        raise RuntimeError("deterministic mode requires an explicit seed")

    if effective_backend == "cuda" and torch.cuda.is_available() is False:
        raise RuntimeError("backend/device mismatch: cuda backend without available cuda device")


def _infer_world_channel_count(world: Any) -> int | None:
    for attr in ("K", "K_channels", "n_channels", "num_channels"):
        value = getattr(world, attr, None)
        if value is None:
            continue
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count

    channels = getattr(world, "channels", None)
    if isinstance(channels, (list, tuple)) and len(channels) > 0:
        return int(len(channels))

    scripted_returns = getattr(world, "_r", None)
    if scripted_returns is not None:
        arr = np.asarray(scripted_returns, dtype=float)
        if arr.ndim == 1 and arr.shape[0] > 0:
            return int(arr.shape[0])

    return None


def _ensure_selector_channels(selector: Any, channels: int) -> None:
    if channels <= 0:
        return
    if hasattr(selector, "ensure_channel_state"):
        selector.ensure_channel_state(int(channels))
        return
    if selector.w is None or len(selector.w) != int(channels):
        selector.w = np.ones(int(channels), dtype=float) / float(max(1, int(channels)))
        selector.K = int(channels)


def _build_runtime_action(selector: Any):
    flow_matrix = getattr(selector, "flow_matrix", None)
    output_weights = getattr(selector, "output_weights", None)
    if flow_matrix is not None:
        fm = np.asarray(flow_matrix, dtype=float)
        if fm.ndim == 2 and fm.shape[0] > 0 and fm.shape[1] > 0:
            out_w = None if output_weights is None else np.asarray(output_weights, dtype=float)
            if out_w is None:
                base_w = selector.allocate() if hasattr(selector, "allocate") else getattr(selector, "w", None)
                if base_w is not None:
                    bw = np.asarray(base_w, dtype=float)
                    if bw.ndim == 1 and bw.shape[0] == int(fm.shape[1]):
                        out_w = bw
            return build_world_action(
                flow_matrix=fm,
                output_weights=out_w,
                expected_channels=int(fm.shape[0]),
            )

    weights = selector.allocate() if hasattr(selector, "allocate") else getattr(selector, "w", None)
    if weights is None:
        return None
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.shape[0] == 0:
        return None
    return build_world_action(weights=w, expected_channels=int(w.shape[0]))


def _world_accepts_action(world: Any) -> bool:
    step_fn = getattr(world, "step", None)
    if not callable(step_fn):
        raise ValueError("world must provide a callable step(...) method")

    try:
        sig = inspect.signature(step_fn)
    except (TypeError, ValueError):
        return False

    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) < 2:
        return False
    return positional[1].name == "action"


def _invoke_world_step(world: Any, *, t: int, action: Any):
    if _world_accepts_action(world):
        return world.step(int(t), action)
    return world.step(int(t))


def run_population(
    *,
    world: World,
    steps: int,
    config: RuntimeConfig,
    backend: str,
    capm_mode: str,
) -> Dict[str, Any]:
    os.environ["CAPM_MODE"] = str(capm_mode)

    selector = (
        CapitalSelectorBuilder()
        .with_K(0)
        .with_selector_policy(config.selector_policy)
        .with_lambda_risk(config.lambda_risk)
        .build()
    )
    manager = PopulationManager.single(
        selector,
        process_id=0,
        backend=str(backend),
        rebirth_config=RebirthConfig(
            enabled=True,
            base_liquidity=float(config.rebirth_base_liquidity),
            eta=float(config.rebirth_eta),
            kappa=float(config.rebirth_kappa),
            max_claims_per_process=int(config.max_claims_per_process),
        ),
    )

    trace: list[str] = []
    history: list[dict[str, Any]] = []
    population_history: list[dict[int, dict[str, Any]]] = []

    for t in range(int(steps)):
        out = world.step(t)

        jackpot = float(out.get("jackpot", 0.0)) if isinstance(out, dict) else 0.0
        process_events: dict[int, dict[str, Any]] = {}

        if isinstance(out, dict) and "population" in out:
            for item in out.get("population", []):
                process_id = int(item["process_id"])
                r_vec = np.asarray(item.get("r", []), dtype=float)
                c_total = float(item.get("c", 0.0))
                process_events[process_id] = {"r_vec": r_vec, "c_total": c_total, "freeze": bool(config.freeze)}
        else:
            r_vec, c_total = validate_world_output(out)
            process_events[0] = {"r_vec": r_vec, "c_total": c_total, "freeze": bool(config.freeze)}

        manager.step_tau(tau=t, process_events=process_events, jackpot=jackpot)

        snapshot = {pid: sel.state() for pid, sel in sorted(manager.processes.items())}
        population_history.append(snapshot)
        history.append(snapshot.get(0, {"wealth": float("nan")}))
        trace.append("step")

    return {
        "history": history,
        "trace": trace,
        "population_history": population_history,
        "inhabitants": manager.inhabitants.entries(),
        "pool": float(manager.burndown.B_current),
        "runtime": {
            "requested_backend": str(backend),
            "effective_backend": str(backend),
            "CAPM_MODE": str(capm_mode),
            "seed": int(config.seed) if config.seed is not None else None,
            "deterministic": bool(config.deterministic),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }


def run(
    *,
    world: World,
    steps: int,
    config: RuntimeConfig | None = None,
    profile: ProfileAConfig | None = None,
) -> Dict[str, Any]:
    """Canonical runtime entry point (Profile A).

    This is a minimal runner for deterministic Profile A semantics.
    """
    cfg = config or RuntimeConfig()
    if cfg.profile != "A":
        raise ValueError("Only Profile A is supported in v1")

    requested_backend, effective_backend = _resolve_backend(cfg)
    effective_mode = _resolve_capm_mode(cfg)
    _validate_builder_runtime(cfg, effective_backend=effective_backend, effective_mode=effective_mode)

    os.environ["CAPM_MODE"] = str(effective_mode)

    if cfg.deterministic:
        enable_determinism(0 if cfg.seed is None else int(cfg.seed))

    prof = profile or ProfileAConfig()
    _ = prof

    if cfg.enable_meta_rebirth:
        return run_population(world=world, steps=steps, config=cfg, backend=effective_backend, capm_mode=effective_mode)

    core = CudaCore() if effective_backend == "cuda" else CpuCore()

    # initialize selector from Profile A defaults
    selector = (
        CapitalSelectorBuilder()
        .with_K(0)
        .with_selector_policy(cfg.selector_policy)
        .with_lambda_risk(cfg.lambda_risk)
        .build()
    )

    trace = []
    history = []
    for t in range(int(steps)):
        hinted_channels = _infer_world_channel_count(world)
        if hinted_channels is not None:
            _ensure_selector_channels(selector, int(hinted_channels))

        action = _build_runtime_action(selector)
        out = _invoke_world_step(world, t=t, action=action)
        r_vec, c_total = validate_world_output(out)
        _ensure_selector_channels(selector, int(len(r_vec)))
        core.step(selector, r_vec, c_total, freeze=cfg.freeze)
        history.append(selector.state())
        trace.append("step")

    return {
        "history": history,
        "trace": trace,
        "runtime": {
            "requested_backend": str(requested_backend),
            "effective_backend": str(effective_backend),
            "CAPM_MODE": str(effective_mode),
            "seed": int(cfg.seed) if cfg.seed is not None else None,
            "deterministic": bool(cfg.deterministic),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    }
