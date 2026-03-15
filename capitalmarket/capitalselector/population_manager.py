from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .builder import build_selector_from_genome
from .cpu_impl import CpuCore
from .cuda_impl import CudaCore
from .evolution_contracts import EDGE_EPSILON
from .evolution_contracts import PAIRWISE_DISTANCE_MAX_EXACT_POPULATION
from .evolution_contracts import PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT
from .evolution_contracts import PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED
from .genome_mutation import compute_mutation_diagnostics
from .genome_mutation import MutationConfig
from .genome_mutation import derive_rebirth_subseeds
from .genome_mutation import mutate_selector_genome
from .genome_mutation import parent_genome_from_entry
from .genome_mutation import select_parent_entry_deterministic
from .genome_mutation import selector_genome_from_selector
from .genome_mutation import serialize_genome_for_metadata
from .inhabitants import InhabitantsBook
from .inhabitants import InhabitantEntry
from .kernel_semantics import step_at_tau
from .lifecycle_cuda import compute_lifecycle_cuda
from .ledger import ClaimLedger
from .parent_selection import build_parent_selection_policy
from .parent_selection import ParentSelectionPolicy
from .mutation_scaling import apply_mutation_scaling
from .mutation_scaling import compute_mutation_scale_factor
from .mutation_scaling import MutationScalingConfig
from .phase_i_state import DEFAULT_LAMBDA_RISK, validate_lambda_risk
from .selector_policy import DEFAULT_SELECTOR_POLICY, validate_selector_policy
from .world_burndown import BurndownPool


CANONICAL_GENERATION_LOOP_SYMBOL_PATH = "capitalmarket.capitalselector.population_manager.run_generation_loop"


def run_generation_loop(
    genomes: Sequence[Any],
    *,
    seed: int | None,
    runtime_horizon: int,
    world_parameters: Mapping[str, Any],
    dt: float = 1.0,
    backend: str = "cpu",
    on_generation_step: Callable[..., None] | None = None,
    on_loop_complete: Callable[..., None] | None = None,
):
    """Canonical v0.9.1 orchestration entrypoint for generation-loop evaluation."""
    if seed is None:
        raise ValueError("generation loop requires deterministic seed")

    resident_genomes = list(genomes)
    if not resident_genomes:
        raise ValueError("generation loop requires at least one genome")

    # Local import avoids circular dependency with fitness_engine.
    from .fitness_engine import (
        FitnessEvaluationConfig,
        build_generation_loop_log_record,
        build_generation_step_log_record,
        evaluate_population_fitness,
    )

    world_config = dict(world_parameters)
    rebirth_defaults = RebirthConfig()
    parent_selection_policy = world_config.get("rebirth_parent_selection_policy", rebirth_defaults.parent_selection_policy)
    if isinstance(parent_selection_policy, str):
        parent_selection_policy = build_parent_selection_policy(parent_selection_policy)

    mutation_scaling_mode = world_config.get("mutation_scaling_mode", None)
    mutation_scaling_config: MutationScalingConfig | None = None
    if mutation_scaling_mode is not None:
        mutation_scaling_config = MutationScalingConfig(
            mode=str(mutation_scaling_mode),
            decay_rate=float(world_config.get("mutation_scaling_decay_rate", 0.01)),
        )

    config = FitnessEvaluationConfig(
        seed=int(seed),
        population_size=int(len(resident_genomes)),
        runtime_horizon=int(runtime_horizon),
        world_parameters=world_config,
        dt=float(dt),
        backend=str(backend),
        evaluation_mode="genome_pipeline",
        rebirth_enabled=bool(world_config.get("rebirth_enabled", False)),
        rebirth_base_liquidity=float(world_config.get("rebirth_base_liquidity", rebirth_defaults.base_liquidity)),
        rebirth_eta=float(world_config.get("rebirth_eta", rebirth_defaults.eta)),
        rebirth_kappa=float(world_config.get("rebirth_kappa", rebirth_defaults.kappa)),
        mutation_noise_scale=float(world_config.get("mutation_noise_scale", rebirth_defaults.mutation_noise_scale)),
        mutation_redistribution_share=float(
            world_config.get("mutation_redistribution_share", rebirth_defaults.mutation_redistribution_share)
        ),
        mutation_lambda_risk_scale=float(
            world_config.get("mutation_lambda_risk_scale", rebirth_defaults.mutation_lambda_risk_scale)
        ),
        rebirth_parent_selection_policy=parent_selection_policy,
        mutation_scaling_config=mutation_scaling_config,
        generation_event_cache_enabled=(
            None
            if "generation_event_cache_enabled" not in world_config
            else bool(world_config.get("generation_event_cache_enabled"))
        ),
        pairwise_distance_mode=str(world_config.get("pairwise_distance_mode", "exact")),
        pairwise_distance_max_exact_population=int(
            world_config.get("pairwise_distance_max_exact_population", PAIRWISE_DISTANCE_MAX_EXACT_POPULATION)
        ),
        pairwise_distance_sampled_pair_count=int(
            world_config.get("pairwise_distance_sampled_pair_count", PAIRWISE_DISTANCE_DEFAULT_SAMPLED_PAIR_COUNT)
        ),
        pairwise_distance_sample_seed=int(world_config.get("pairwise_distance_sample_seed", PAIRWISE_DISTANCE_DEFAULT_SAMPLE_SEED)),
    )
    report = evaluate_population_fitness(resident_genomes, config)

    if on_generation_step is not None:
        for step in report.generation_trajectory.steps:
            on_generation_step(build_generation_step_log_record(step))

    if on_loop_complete is not None:
        on_loop_complete(build_generation_loop_log_record(report))

    return report


@dataclass(frozen=True)
class RebirthConfig:
    enabled: bool = True
    base_liquidity: float = 0.0
    eta: float = 0.0
    kappa: float = 1.0
    selection_epsilon: float = EDGE_EPSILON
    deterministic_seed: int | None = 0
    mutation_noise_scale: float = 0.01
    mutation_redistribution_share: float = 0.05
    mutation_lambda_risk_scale: float = 0.05
    max_claims_per_process: int = 1_000_000
    parent_selection_policy: ParentSelectionPolicy | None = None
    mutation_scaling_config: MutationScalingConfig | None = None


class PopulationManager:
    """CPU meta-layer for dead archival and rebirth instantiation."""

    def __init__(self, *, processes: dict[int, Any], rebirth_config: RebirthConfig | None = None, backend: str = "cpu"):
        self.processes: dict[int, Any] = dict(processes)
        self.rebirth_config = rebirth_config or RebirthConfig()
        self.backend = str(backend)
        self.inhabitants = InhabitantsBook()
        self.burndown = BurndownPool()
        self._next_process_id = (max(self.processes.keys()) + 1) if self.processes else 0
        self._cores: dict[int, Any] = {}

        for process_id, selector in self.processes.items():
            self._ensure_selector_meta(selector, process_id)
            self._cores[process_id] = self._build_core(start_tau=0)

    @classmethod
    def single(cls, selector: Any, *, process_id: int = 0, rebirth_config: RebirthConfig | None = None, backend: str = "cpu"):
        manager = cls(processes={int(process_id): selector}, rebirth_config=rebirth_config, backend=backend)
        return manager

    def step_tau(self, *, tau: int, process_events: Mapping[int, Mapping[str, Any]], jackpot: float = 0.0) -> dict[str, Any]:
        dead_now: list[Any] = []
        dead_keys: list[tuple[int, int]] = []
        active_cuda_selectors: list[Any] = []
        active_cuda_cores: list[Any] = []

        for process_id in sorted(self.processes.keys()):
            selector = self.processes[process_id]
            if bool(getattr(selector, "dead", False)):
                continue

            event = dict(process_events.get(process_id, {}))
            r_vec = np.asarray(event.get("r_vec", []), dtype=float)
            c_total = float(event.get("c_total", 0.0))
            freeze = bool(event.get("freeze", False))

            if hasattr(selector, "ensure_channel_state"):
                selector.ensure_channel_state(len(r_vec))
            elif selector.w is None or len(selector.w) != len(r_vec):
                selector.w = np.ones(len(r_vec)) / max(1, len(r_vec))
                selector.K = len(r_vec)

            core = self._cores[process_id]
            if hasattr(core, "step_with_tau"):
                core.step_with_tau(selector, r_vec, c_total, freeze=freeze, tau=int(tau))
            else:
                step_at_tau(
                    selector,
                    {"r_vec": r_vec, "c_total": c_total, "freeze": freeze},
                    policy=None,
                    tau=int(tau),
                    hooks=None,
                )

            selector._fitness_integral = float(getattr(selector, "_fitness_integral", 0.0)) + float(selector.wealth)

            if self.backend == "cuda" and hasattr(core, "lifecycle_snapshot"):
                active_cuda_selectors.append(selector)
                active_cuda_cores.append(core)
            else:
                is_dead = bool(getattr(selector, "_last_settlement_failed", False)) or float(selector.wealth) < 0.0
                if is_dead:
                    selector.dead = True
                    selector.tau_dead = int(tau)
                    dead_now.append(selector)
                    dead_keys.append((int(selector.process_id), int(selector.generation_id)))

        allocations_by_pid: dict[int, float] = {}
        if self.backend == "cuda" and active_cuda_selectors:
            snaps = [core.lifecycle_snapshot(selector) for selector, core in zip(active_cuda_selectors, active_cuda_cores)]
            wealth_t = torch.cat([snap["wealth"] for snap in snaps], dim=0)
            dead_t = torch.cat([snap["dead_mask"] for snap in snaps], dim=0)
            pid_t = torch.cat([snap["process_id"] for snap in snaps], dim=0)
            gen_t = torch.cat([snap["generation_id"] for snap in snaps], dim=0)
            dead_keys_t = torch.stack([pid_t, gen_t], dim=1)

            lifecycle = compute_lifecycle_cuda(
                wealth=wealth_t,
                dead_mask_semantic=dead_t,
                dead_keys=dead_keys_t,
                pool_before=float(self.burndown.B_current),
                jackpot=float(jackpot),
                rebirth_enabled=bool(self.rebirth_config.enabled),
                base_liquidity=float(self.rebirth_config.base_liquidity),
                eta=float(self.rebirth_config.eta),
                kappa=float(self.rebirth_config.kappa),
                epsilon=float(self.rebirth_config.selection_epsilon),
            )

            self.burndown.B_current = float(lifecycle.pool_final.item())
            dead_idx = torch.nonzero(lifecycle.dead_mask, as_tuple=False).flatten().tolist()

            for dead_offset, idx in enumerate(dead_idx):
                selector = active_cuda_selectors[int(idx)]
                selector.dead = True
                selector.tau_dead = int(tau)
                dead_now.append(selector)
                dead_keys.append((int(selector.process_id), int(selector.generation_id)))
                if int(dead_offset) < int(lifecycle.rebirth_allocations_dead.shape[0]):
                    allocations_by_pid[int(selector.process_id)] = float(lifecycle.rebirth_allocations_dead[int(dead_offset)].item())
        else:
            burn_total = float(sum(max(0.0, -float(selector.wealth)) for selector in dead_now))
            self.burndown.apply_tau_inflows(burn=burn_total, kappa=self.rebirth_config.kappa, jackpot=jackpot)

        dead_now.sort(key=lambda selector: (int(selector.process_id), int(selector.generation_id)))
        for selector in dead_now:
            parent_genome = selector_genome_from_selector(selector)
            self.inhabitants.append_dead(
                process_id=int(selector.process_id),
                generation_id=int(selector.generation_id),
                tau_dead=int(selector.tau_dead),
                fitness=float(getattr(selector, "_fitness_integral", 0.0)),
                final_liquidity=float(selector.wealth),
                metadata={
                    "kind": str(getattr(selector, "kind", "unknown")),
                    "rebirth_threshold": float(getattr(selector, "rebirth_threshold", 0.0)),
                    "genome": serialize_genome_for_metadata(parent_genome),
                },
            )

        newborn_ids: list[int] = []
        mutation_diagnostics: list[dict[str, Any]] = []
        if self.rebirth_config.enabled and dead_now:
            if self.backend == "cuda":
                allocations = [float(allocations_by_pid.get(int(selector.process_id), 0.0)) for selector in dead_now]
            else:
                B_tau = float(self.burndown.B_current)
                requested = [float(self.rebirth_config.base_liquidity + self.rebirth_config.eta * B_tau) for _ in dead_now]
                allocations = self.burndown.allocate_fair_same_tau(
                    requested=requested,
                    stable_keys=dead_keys,
                    epsilon=self.rebirth_config.selection_epsilon,
                )

            for selector, allocation in zip(dead_now, allocations):
                selection_seed, mutation_seed = derive_rebirth_subseeds(
                    deterministic_seed=self.rebirth_config.deterministic_seed,
                    tau=int(tau),
                    dead_process_id=int(selector.process_id),
                    dead_generation_id=int(selector.generation_id),
                )
                parent = self._select_parent_for_rebirth(selection_seed=selection_seed)
                parent_genome = parent_genome_from_entry(parent)
                child_generation_id = int(parent.generation_id + 1)
                _base_mutation_config = MutationConfig(
                    noise_scale=float(self.rebirth_config.mutation_noise_scale),
                    redistribution_share=float(self.rebirth_config.mutation_redistribution_share),
                    lambda_risk_scale=float(self.rebirth_config.mutation_lambda_risk_scale),
                )
                if self.rebirth_config.mutation_scaling_config is not None:
                    _scale_factor = compute_mutation_scale_factor(
                        generation_index=int(child_generation_id),
                        config=self.rebirth_config.mutation_scaling_config,
                    )
                    _effective_mutation_config = apply_mutation_scaling(
                        _base_mutation_config, scale_factor=_scale_factor
                    )
                else:
                    _effective_mutation_config = _base_mutation_config
                newborn_genome = mutate_selector_genome(
                    parent_genome,
                    mutation_seed=mutation_seed,
                    config=_effective_mutation_config,
                    execution_backend="cpu",
                )
                diagnostics = compute_mutation_diagnostics(
                    parent_flow_matrix=np.asarray(parent_genome.flow_matrix, dtype=np.float64),
                    child_flow_matrix=np.asarray(newborn_genome.flow_matrix, dtype=np.float64),
                )
                generation_id = int(child_generation_id)
                new_id = self._allocate_process_id()
                newborn = self._instantiate_new_process(
                    process_id=new_id,
                    generation_id=generation_id,
                    liquidity=float(allocation),
                    parent_entry=parent,
                    genome=newborn_genome,
                )
                newborn._last_mutation_diagnostics = {
                    "mutation_magnitude": float(diagnostics.mutation_magnitude),
                    "modified_edge_count": int(diagnostics.modified_edge_count),
                    "structural_distance": float(diagnostics.structural_distance),
                }
                mutation_diagnostics.append(
                    {
                        "newborn_process_id": int(new_id),
                        "parent_process_id": int(parent.process_id),
                        "parent_generation_id": int(parent.generation_id),
                        "mutation_magnitude": float(diagnostics.mutation_magnitude),
                        "modified_edge_count": int(diagnostics.modified_edge_count),
                        "structural_distance": float(diagnostics.structural_distance),
                    }
                )
                self.processes[new_id] = newborn
                self._cores[new_id] = self._build_core(start_tau=int(tau) + 1)
                newborn_ids.append(new_id)

        return {
            "dead_ids": [int(selector.process_id) for selector in dead_now],
            "newborn_ids": newborn_ids,
            "mutation_diagnostics": mutation_diagnostics,
            "pool": float(self.burndown.B_current),
        }

    def _ordered_inhabitant_entries(self) -> list[InhabitantEntry]:
        entries = self.inhabitants.entries()
        return sorted(entries, key=lambda entry: (int(entry.process_id), int(entry.generation_id)))

    def _selection_rng_substream(self, *, selection_seed: int | None) -> np.random.Generator:
        if selection_seed is None:
            raise ValueError("parent selection requires deterministic seed")
        seed_i = int(selection_seed)
        return np.random.default_rng(np.random.PCG64(np.uint64(seed_i)))

    def _select_parent_for_rebirth(self, *, selection_seed: int | None) -> InhabitantEntry:
        ordered_entries = self._ordered_inhabitant_entries()
        policy = self.rebirth_config.parent_selection_policy
        if policy is None:
            return select_parent_entry_deterministic(
                ordered_entries,
                selection_seed=selection_seed,
                epsilon=float(self.rebirth_config.selection_epsilon),
            )

        rng = self._selection_rng_substream(selection_seed=selection_seed)
        fitness = np.asarray([float(entry.fitness) for entry in ordered_entries], dtype=np.float64)
        parent_indices = policy.select_parents(
            population=ordered_entries,
            fitness=fitness,
            n_parents=1,
            rng=rng,
        )
        if len(parent_indices) == 0:
            raise ValueError("parent selection policy returned empty index set")

        parent_index = int(parent_indices[0])
        if parent_index < 0 or parent_index >= len(ordered_entries):
            raise ValueError("parent selection policy returned out-of-bounds index")
        return ordered_entries[parent_index]

    def _instantiate_new_process(
        self,
        *,
        process_id: int,
        generation_id: int,
        liquidity: float,
        parent_entry: InhabitantEntry,
        genome: Any,
    ):
        metadata = dict(parent_entry.metadata or {})
        kind = str(metadata.get("kind", "entrepreneur"))
        rebirth_threshold = float(metadata.get("rebirth_threshold", 0.0))

        selector = build_selector_from_genome(
            genome,
            process_id=int(process_id),
            generation_id=int(generation_id),
            initial_wealth=float(liquidity),
            rebirth_threshold=rebirth_threshold,
            kind=kind,
        )

        expected_flow = np.asarray(genome.flow_matrix, dtype=float)
        actual_flow = np.asarray(getattr(selector, "flow_matrix", []), dtype=float)
        expected_output = np.asarray(genome.output_weights, dtype=float)
        actual_output = np.asarray(getattr(selector, "output_weights", []), dtype=float)
        if expected_flow.shape != actual_flow.shape or not np.array_equal(expected_flow, actual_flow):
            raise ValueError("rebirth bypassed genome inheritance")
        if expected_output.shape != actual_output.shape or not np.array_equal(expected_output, actual_output):
            raise ValueError("rebirth bypassed genome inheritance")
        if float(getattr(selector, "lambda_risk", float("nan"))) != float(genome.lambda_risk):
            raise ValueError("rebirth bypassed genome inheritance")
        if str(getattr(selector, "selector_policy", "")) != str(genome.selector_policy):
            raise ValueError("rebirth bypassed genome inheritance")

        selector.process_id = int(process_id)
        selector.generation_id = int(generation_id)
        selector.parent_process_id = int(parent_entry.process_id)
        selector.parent_generation_id = int(parent_entry.generation_id)
        selector.dead = False
        selector.tau_dead = None
        selector._fitness_integral = 0.0
        selector.claim_ledger = ClaimLedger(max_claims_per_process=self.rebirth_config.max_claims_per_process)
        selector.offers = []
        return selector

    def _allocate_process_id(self) -> int:
        process_id = int(self._next_process_id)
        self._next_process_id += 1
        return process_id

    def _ensure_selector_meta(self, selector: Any, process_id: int) -> None:
        required_accounting = ("liquidity", "claim_ledger", "offers", "dead", "dead_flag", "tau_dead", "generation_id")
        missing = [name for name in required_accounting if not hasattr(selector, name)]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"selector missing canonical accounting core fields: {joined}")

        selector.process_id = int(getattr(selector, "process_id", process_id))
        selector.generation_id = int(getattr(selector, "generation_id"))
        selector.dead = bool(getattr(selector, "dead"))
        selector.dead_flag = bool(getattr(selector, "dead_flag"))
        selector.tau_dead = getattr(selector, "tau_dead")
        selector._fitness_integral = float(getattr(selector, "_fitness_integral", 0.0))
        selector.selector_policy = validate_selector_policy(str(getattr(selector, "selector_policy", DEFAULT_SELECTOR_POLICY)))
        selector.policy = selector.selector_policy
        selector.lambda_risk = validate_lambda_risk(float(getattr(selector, "lambda_risk", DEFAULT_LAMBDA_RISK)))
        selector.liquidity = float(getattr(selector, "liquidity"))
        selector.settlement_config = dict(getattr(selector, "settlement_config", {}) or {})

        ledger = getattr(selector, "claim_ledger")
        if not isinstance(ledger, ClaimLedger):
            raise ValueError("selector.claim_ledger must be ClaimLedger")

        offers = list(getattr(selector, "offers") or [])
        selector.offers = offers

    def _build_core(self, *, start_tau: int):
        if self.backend == "cuda":
            return CudaCore(start_tau=int(start_tau), device="cuda")
        return CpuCore(start_tau=int(start_tau))
