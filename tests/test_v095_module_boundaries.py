from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import runpy
import subprocess
from types import ModuleType
import warnings

from capitalmarket.capitalselector.ess_evaluator import ESSProbeConfig
from capitalmarket.capitalselector.experiments.long_run_harness import LongRunConfig
from capitalmarket.capitalselector.experiments.long_run_harness import run_long_run_harness
from capitalmarket.capitalselector.regime_robustness import RegimeRobustnessConfig


PACKAGE_PREFIX = "capitalmarket.capitalselector"
PACKAGE_ROOT = Path("capitalmarket/capitalselector")
PUBLIC_API_FILE = PACKAGE_ROOT / "__init__.py"
PUBLIC_API_MODULE = "capitalmarket.capitalselector"
V094_PUBLIC_API_BASELINE_FILE = Path("tests/baselines/v094_public_api_all.json")

RUNTIME_CORE_EXPLICIT_MODULES = {
    f"{PACKAGE_PREFIX}.world_interface",
    f"{PACKAGE_PREFIX}.world_parameters",
    f"{PACKAGE_PREFIX}.evolution_contracts",
    f"{PACKAGE_PREFIX}.worlds",
}
EVOLUTION_ENGINE_MODULES = {
    f"{PACKAGE_PREFIX}.fitness_engine",
    f"{PACKAGE_PREFIX}.genome",
    f"{PACKAGE_PREFIX}.population_manager",
    f"{PACKAGE_PREFIX}.parent_selection",
    f"{PACKAGE_PREFIX}.mutation_scaling",
}
EXPERIMENT_LAYER_WRAPPERS = {
    f"{PACKAGE_PREFIX}.ess_evaluator",
    f"{PACKAGE_PREFIX}.periodicity_experiment",
    f"{PACKAGE_PREFIX}.regime_robustness",
    f"{PACKAGE_PREFIX}.experiments",
}


def _module_name_from_path(path: Path) -> str:
    module_name = path.with_suffix("").as_posix().replace("/", ".")
    if module_name.endswith(".__init__"):
        module_name = module_name[: -len(".__init__")]
    return module_name


def _iter_module_files() -> list[Path]:
    return sorted(path for path in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _parse_public_symbols_from_source(source: str) -> list[str]:
    module = ast.parse(source)
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if not isinstance(node.value, (ast.List, ast.Tuple)):
                    raise AssertionError("__all__ must be an explicit list/tuple of strings")
                out: list[str] = []
                for item in node.value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        out.append(item.value)
                    else:
                        raise AssertionError("__all__ must contain only string literals")
                return out
    raise AssertionError("public API __all__ declaration not found")


def _parse_public_symbols_from_file(path: Path) -> list[str]:
    return _parse_public_symbols_from_source(path.read_text(encoding="utf-8"))


def _declared_top_level_candidates_from_source(source: str) -> set[str]:
    module = ast.parse(source)
    candidates: set[str] = set()

    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                if not name.startswith("_"):
                    candidates.add(name)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                candidates.add(node.name)
            continue

    candidates.discard("__all__")
    return candidates


def _parse_public_symbols_from_git_ref(git_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "show", f"{git_ref}:{PUBLIC_API_FILE.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        if not V094_PUBLIC_API_BASELINE_FILE.exists():
            raise AssertionError(
                f"unable to read public API snapshot from git ref {git_ref} and missing baseline "
                f"{V094_PUBLIC_API_BASELINE_FILE.as_posix()}"
            )

        payload = json.loads(V094_PUBLIC_API_BASELINE_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
            raise AssertionError("v0.9.4 public API baseline must be a JSON list of strings")
        return [str(item) for item in payload]

    return _parse_public_symbols_from_source(result.stdout)


def _normalize_to_known_module(name: str, module_names: set[str]) -> str | None:
    if not name.startswith(PACKAGE_PREFIX):
        return None

    candidate = name
    while True:
        if candidate in module_names:
            return candidate
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]
    return name


def _resolve_import_targets(source_module: str, node: ast.Import | ast.ImportFrom, module_names: set[str]) -> set[str]:
    targets: set[str] = set()

    if isinstance(node, ast.Import):
        for alias in node.names:
            resolved = _normalize_to_known_module(alias.name, module_names)
            if resolved is not None:
                targets.add(resolved)
        return targets

    source_parts = source_module.split(".")
    if node.level > 0:
        if node.level > len(source_parts):
            return targets
        base_parts = source_parts[: -node.level]
    else:
        base_parts = []

    if node.module:
        target_base = ".".join(base_parts + node.module.split("."))
    else:
        target_base = ".".join(base_parts)

    resolved_base = _normalize_to_known_module(target_base, module_names)
    if resolved_base is not None:
        targets.add(resolved_base)

    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = f"{target_base}.{alias.name}" if target_base else alias.name
        resolved_candidate = _normalize_to_known_module(candidate, module_names)
        if resolved_candidate is not None:
            targets.add(resolved_candidate)

    return targets


def _collect_import_graph() -> tuple[set[str], dict[str, set[str]]]:
    module_files = _iter_module_files()
    module_names = {_module_name_from_path(path) for path in module_files}
    graph: dict[str, set[str]] = {module_name: set() for module_name in module_names}

    for path in module_files:
        source_module = _module_name_from_path(path)
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                graph[source_module].update(_resolve_import_targets(source_module, node, module_names))

    return module_names, graph


def _layer_for_module(module_name: str) -> str | None:
    if module_name in RUNTIME_CORE_EXPLICIT_MODULES or module_name.startswith(f"{PACKAGE_PREFIX}.worlds."):
        return "runtime"
    if module_name in EVOLUTION_ENGINE_MODULES:
        return "evolution"
    if module_name in EXPERIMENT_LAYER_WRAPPERS or module_name.startswith(f"{PACKAGE_PREFIX}.experiments."):
        return "experiment"
    if module_name == f"{PACKAGE_PREFIX}.metrics" or module_name.startswith(f"{PACKAGE_PREFIX}.metrics."):
        return "metrics"
    return None


def _modules_in_layer(module_names: set[str], layer_name: str) -> set[str]:
    return {name for name in module_names if _layer_for_module(name) == layer_name}


def _strongly_connected_components(nodes: set[str], graph: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    node_index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        node_index[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph.get(node, set()):
            if target not in nodes:
                continue
            if target not in node_index:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], node_index[target])

        if lowlink[node] == node_index[node]:
            component: list[str] = []
            while True:
                item = stack.pop()
                on_stack.remove(item)
                component.append(item)
                if item == node:
                    break
            components.append(sorted(component))

    for node in sorted(nodes):
        if node not in node_index:
            visit(node)

    return components


def _component_trace(component: list[str], graph: dict[str, set[str]]) -> list[str]:
    component_set = set(component)
    edges: list[str] = []
    for source in sorted(component):
        for target in sorted(graph.get(source, set())):
            if target in component_set:
                edges.append(f"{source} -> {target}")
    return edges


def test_v095_module_boundaries_runtime_core_has_no_experiment_or_metrics_imports() -> None:
    module_names, graph = _collect_import_graph()
    runtime_modules = _modules_in_layer(module_names, "runtime")

    violations: list[str] = []
    for source in sorted(runtime_modules):
        for target in sorted(graph.get(source, set())):
            layer = _layer_for_module(target)
            if layer in {"experiment", "metrics"}:
                violations.append(f"{source} -> {target}")

    assert not violations, "runtime core must not import experiment/metrics modules:\n" + "\n".join(violations)


def test_v095_module_boundaries_no_cross_layer_circular_dependencies() -> None:
    module_names, graph = _collect_import_graph()
    layer_pairs = [
        ("runtime", "evolution"),
        ("evolution", "experiment"),
        ("experiment", "metrics"),
    ]

    violations: list[str] = []
    for layer_a, layer_b in layer_pairs:
        nodes = {
            module_name
            for module_name in module_names
            if _layer_for_module(module_name) in {layer_a, layer_b}
        }
        if len(nodes) <= 1:
            continue

        components = _strongly_connected_components(nodes, graph)
        for component in components:
            if len(component) <= 1:
                continue
            component_layers = {_layer_for_module(module_name) for module_name in component}
            if layer_a in component_layers and layer_b in component_layers:
                trace_lines = _component_trace(component, graph)
                violations.append(
                    f"pair={layer_a}<->{layer_b} component={component} trace={trace_lines}"
                )

    assert not violations, "static import trace found cross-layer circular dependencies:\n" + "\n".join(violations)


def test_v095_module_boundaries_public_api_symbols_are_individually_importable() -> None:
    public_symbols = _parse_public_symbols_from_file(PUBLIC_API_FILE)
    assert public_symbols, "public API __all__ must not be empty"
    assert len(public_symbols) == len(set(public_symbols)), "public API __all__ must not contain duplicates"

    imported: list[str] = []
    for symbol in public_symbols:
        namespace: dict[str, object] = {}
        exec(f"from {PUBLIC_API_MODULE} import {symbol}", namespace, namespace)
        assert symbol in namespace
        imported.append(symbol)

    assert len(imported) == len(public_symbols)


def test_v095_module_boundaries_no_implicit_public_api_surface() -> None:
    public_symbols = set(_parse_public_symbols_from_file(PUBLIC_API_FILE))
    source = PUBLIC_API_FILE.read_text(encoding="utf-8")
    declared_candidates = _declared_top_level_candidates_from_source(source)

    undeclared_declared_candidates = sorted(declared_candidates - public_symbols)
    assert not undeclared_declared_candidates, (
        "top-level names imported/defined in __init__.py must be explicitly listed in __all__: "
        + str(undeclared_declared_candidates)
    )

    module = importlib.import_module(PUBLIC_API_MODULE)
    runtime_public_names = {
        name
        for name in module.__dict__.keys()
        if not name.startswith("_") and not isinstance(module.__dict__[name], ModuleType)
    }
    implicit_runtime_public_names = sorted(runtime_public_names - public_symbols)
    assert not implicit_runtime_public_names, (
        "implicit runtime-visible public names are forbidden (must be declared in __all__): "
        + str(implicit_runtime_public_names)
    )


def test_v095_module_boundaries_removed_symbols_have_deprecation_wrappers() -> None:
    v094_public_symbols = set(_parse_public_symbols_from_git_ref("v0.9.4"))
    current_public_symbols = set(_parse_public_symbols_from_file(PUBLIC_API_FILE))
    removed_symbols = sorted(v094_public_symbols - current_public_symbols)
    api_module = importlib.import_module(PUBLIC_API_MODULE)
    replacements = getattr(api_module, "DEPRECATED_PUBLIC_API_REPLACEMENTS", None)

    if replacements is not None:
        assert isinstance(replacements, dict), "DEPRECATED_PUBLIC_API_REPLACEMENTS must be a dict when present"

    if len(removed_symbols) == 0:
        if isinstance(replacements, dict):
            stale_entries = sorted(symbol for symbol in replacements.keys() if symbol not in removed_symbols)
            assert not stale_entries, (
                "deprecation replacement map must not contain stale symbols unrelated to v0.9.4 baseline: "
                + str(stale_entries)
            )
        assert removed_symbols == []
        return

    assert isinstance(replacements, dict), "DEPRECATED_PUBLIC_API_REPLACEMENTS must map removed symbols to replacements"

    missing_replacements = [symbol for symbol in removed_symbols if symbol not in replacements]
    assert not missing_replacements, "removed public API symbols missing replacement map entries: " + str(missing_replacements)

    for symbol in removed_symbols:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", DeprecationWarning)
            namespace: dict[str, object] = {}
            exec(f"from {PUBLIC_API_MODULE} import {symbol}", namespace, namespace)

        deprecation_messages = [
            str(item.message)
            for item in captured
            if issubclass(item.category, DeprecationWarning)
        ]
        assert deprecation_messages, f"missing DeprecationWarning for removed symbol {symbol}"
        replacement = str(replacements[symbol])
        assert replacement.strip() != "", f"replacement path for {symbol} must not be empty"
        assert "." in replacement, f"replacement path for {symbol} must be a dotted import path"
        assert replacement in deprecation_messages[0], (
            f"deprecation message for {symbol} must reference replacement path {replacement}"
        )


def test_v095_module_boundaries_ag3_v094_configs_instantiate_without_error() -> None:
    legacy_ess_tests = runpy.run_path("tests/test_v094_ess_probe_experiment.py")
    legacy_regime_tests = runpy.run_path("tests/test_v094_regime_robustness.py")

    ess_builder = legacy_ess_tests.get("_probe_config")
    assert callable(ess_builder)
    ess_config = ess_builder(seed=17, backend="cpu")
    assert isinstance(ess_config, ESSProbeConfig)

    regime_builder = legacy_regime_tests.get("_config")
    assert callable(regime_builder)
    regime_config = regime_builder(seed=41, backend="cpu")
    assert isinstance(regime_config, RegimeRobustnessConfig)

    long_run_payload = json.loads(Path("tests/baselines/v094_long_run_config.json").read_text(encoding="utf-8"))

    long_run_payload_for_object = dict(long_run_payload)
    long_run_payload_for_object.pop("config_version", None)
    long_run_payload_for_object["horizon"] = int(long_run_payload_for_object.pop("runtime_horizon"))
    long_run_payload_for_object.setdefault("channel_count", None)
    long_run_payload_for_object.setdefault("generations", None)
    long_run_config = LongRunConfig(**long_run_payload_for_object)
    assert isinstance(long_run_config, LongRunConfig)

    result = run_long_run_harness(config=long_run_payload)
    assert len(result.generation_records) == 1
