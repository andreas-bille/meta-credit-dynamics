"""Fitness distribution visualization utility (B2).

Produces deterministic fitness distribution arrays from ``ExperimentDataset``
inputs.  No plotting or rendering library is used; outputs are plain numpy
arrays suitable for passing to any renderer.

Public API
----------
build_fitness_distribution_dataset(dataset) -> np.ndarray
    Returns a 2-D array of shape ``(n_generations, 2)`` where column 0 is
    ``generation_index`` and column 1 is ``invasion_rate`` (used as a proxy
    for fitness spread across resident generations).  Output is sorted by
    ``generation_index`` and is fully deterministic for a given dataset.
"""

from __future__ import annotations

import numpy as np

from ..experiment_dataset import ExperimentDataset


def build_fitness_distribution_dataset(dataset: ExperimentDataset) -> np.ndarray:
    """Build a deterministic fitness distribution array from an experiment.

    Parameters
    ----------
    dataset:
        Source ``ExperimentDataset``.  ``generation_metrics`` must be
        non-empty; order is not assumed (output is sorted by
        ``generation_index``).

    Returns
    -------
    np.ndarray
        Shape ``(n_generations, 2)``.  Column 0: ``generation_index``
        (float). Column 1: ``invasion_rate`` (float).  Rows are sorted in
        ascending ``generation_index`` order.
    """
    rows = sorted(dataset.generation_metrics, key=lambda gm: gm.generation_index)
    return np.array(
        [[float(gm.generation_index), float(gm.invasion_rate)] for gm in rows],
        dtype=float,
    )
