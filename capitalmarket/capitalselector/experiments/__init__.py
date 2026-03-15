"""Legacy research experiments.

These experiments predate the v0.9.x canonical evolution runner
(run_generation_loop) and intentionally operate at a lower
runtime abstraction level.

They are retained for reproducibility of earlier phases.
"""

from .g3_3_sweep import run_g3_3_sweep
from .experiment_dataset import EXPERIMENT_DATASET_SCHEMA_VERSION
from .experiment_dataset import ExperimentDataset
from .experiment_dataset import GenerationMetrics
from .experiment_dataset import PopulationStatistics
from .experiment_dataset import StrategyMetrics
from .experiment_dataset import WorldRegimeEntry
from .experiment_dataset import ess_dataset_to_experiment_dataset
from .experiment_dataset import experiment_dataset_from_dict
from .experiment_dataset import experiment_dataset_to_dict
from .long_run_harness import LongRunConfig
from .long_run_harness import LongRunGenerationRecord
from .long_run_harness import LongRunEmergenceEvaluation
from .long_run_harness import LongRunResult
from .long_run_harness import LONG_RUN_CHECKPOINT_FILE_STEM_TEMPLATE
from .long_run_harness import LONG_RUN_CHECKPOINT_SCHEMA_VERSION
from .long_run_harness import load_long_run_checkpoint
from .long_run_harness import resolve_long_run_checkpoint_file
from .long_run_harness import evaluate_long_run_emergence_metrics
from .long_run_harness import run_long_run_harness


def run_phase_i_evaluation(*args, **kwargs):
    """Lazy import to avoid runpy warnings for `python -m ...run_phase_i`."""
    from .run_phase_i import run_phase_i_evaluation as _run_phase_i_evaluation

    return _run_phase_i_evaluation(*args, **kwargs)


def run_phase_ii_episode(*args, **kwargs):
    """Lazy import additive closed-loop runner for Phase II."""
    from .run_phase_ii import run_phase_ii_episode as _run_phase_ii_episode

    return _run_phase_ii_episode(*args, **kwargs)


def run_phase_ii_evaluation(*args, **kwargs):
    """Lazy import Phase-II paired-bootstrap evaluation protocol."""
    from .phase_ii_evaluation import run_phase_ii_evaluation as _run_phase_ii_evaluation

    return _run_phase_ii_evaluation(*args, **kwargs)

__all__ = [
    "LONG_RUN_CHECKPOINT_SCHEMA_VERSION",
    "LONG_RUN_CHECKPOINT_FILE_STEM_TEMPLATE",
    "LongRunConfig",
    "LongRunGenerationRecord",
    "LongRunEmergenceEvaluation",
    "LongRunResult",
    "evaluate_long_run_emergence_metrics",
    "load_long_run_checkpoint",
    "resolve_long_run_checkpoint_file",
    "run_g3_3_sweep",
    "run_long_run_harness",
    "run_phase_i_evaluation",
    "run_phase_ii_episode",
    "run_phase_ii_evaluation",
    "EXPERIMENT_DATASET_SCHEMA_VERSION",
    "ExperimentDataset",
    "GenerationMetrics",
    "PopulationStatistics",
    "StrategyMetrics",
    "WorldRegimeEntry",
    "ess_dataset_to_experiment_dataset",
    "experiment_dataset_from_dict",
    "experiment_dataset_to_dict",
]
