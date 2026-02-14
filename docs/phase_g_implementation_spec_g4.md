# Phase G -- G4 CUDA & PyTorch Implementierungsplan (v0.6.4-alpha)

------------------------------------------------------------------------

## Überblick

Ziel: Saubere CUDA-Migration mit vollständiger Testabdeckung und
semantischer Parität zur CPU-Referenz (Phase F / G3).

Prinzip: - Jeder Implementierungsschritt ist durch Tests abgesichert. -
Kein semantischer Drift. - CPU bleibt Ground Truth.

Determinismus: CUDA‑Tests laufen mit `torch.use_deterministic_algorithms(True)` und
`torch.backends.cudnn.benchmark = False` (vgl. concept_g4_v0).

------------------------------------------------------------------------

# STEP 0 --- Safety Harness

## Ziel

CPU-Verhalten einfrieren als Referenz.

## Maßnahmen

### 0.1 Snapshot-Test einführen

Neue Datei:

    tests/test_reference_cpu_phase_g.py

Test: - Seed 0 - 500 Steps - Profile A - Regime Switch World (Topology off)

Assertions: - final wealth - final weights - final cum_pi - final
drawdown - final top2 share (sum of top-2 weights)

Hinweis: Topology explizit deaktivieren (z.B. CAPM_ENABLE_TOPOLOGY=0) für stabile Referenz.

Referenzwerte fest im Test speichern.

### Stop Criterion

CPU ist eingefroren. Alle Tests grün.

------------------------------------------------------------------------

# STEP 1 --- DeviceState Einführung

## Ziel

State-Container tensorisieren (noch CPU-only).

Bestehende Datei erweitern:

    capitalmarket/capitalselector/cuda_state.py

Struktur:

``` python
@dataclass
class DeviceState:
    weights: torch.Tensor
    wealth: torch.Tensor
    mean: torch.Tensor
    var: torch.Tensor
    drawdown: torch.Tensor
    cum_pi: torch.Tensor
    peak_cum_pi: torch.Tensor
    rebirth_threshold: torch.Tensor

    def to(self, device):
        ...
```

CPU → Tensor Adapter in `capitalmarket/capitalselector/cuda_state.py`:

``` python
def to_device_state(self) -> DeviceState:
    ...
```

Batch-Dimension mit `unsqueeze(0)` (B=1).

Test:

    tests/test_device_state_roundtrip_cpu.py

Assertion: dtype-abhängig (float64 strenger als float32)

Zusätzlich: Round‑trip State‑Dump (CPU → CUDA → CPU) aus G2 wiederverwenden.

### Stop Criterion

CPU-only Roundtrip identisch.

------------------------------------------------------------------------

# STEP 2 --- CUDA Backend Skeleton

Neue Dateien (unter `capitalmarket/capitalselector/`):

    cpu_impl.py
    cuda_impl.py

CudaCore Skeleton:

``` python
class CudaCore:
    def step(self, state: DeviceState):
        raise NotImplementedError
```

Runtime Switch im bestehenden Builder/Entry:

``` python
In `capitalmarket/capitalselector/builder.py` (CapitalSelectorBuilder.build),
oder im zentralen Entry‑Point (z.B. `capitalmarket/capitalselector/runtime.py`):

    if device.type == "cuda":
        core = CudaCore()
    else:
        core = CpuCore()

Keine Verteilung von `device.type` Checks im Kerncode (nur hier).
```

Test:

    tests/test_backend_switch_cpu.py

### Stop Criterion

Backend-Switch ändert nichts.

------------------------------------------------------------------------

# STEP 3 --- Reweight CUDA Port

Implementation:

``` python
with torch.no_grad():
    w = state.weights
    scores = ...
    w = w * torch.exp(eta * scores)
    w = w / w.sum(dim=1, keepdim=True)
    state.weights = w
```

Test:

    tests/test_cuda_reweight_parity.py

Tolerance: dtype-abhängig (float64 strenger als float32)

### Stop Criterion

Single-Step Reweight identisch.

------------------------------------------------------------------------

# STEP 4 --- EWMA Port

Implementation:

``` python
mean_prev = state.mean.clone()
state.mean = alpha * returns + (1-alpha) * state.mean
state.var  = alpha * (returns - mean_prev)**2 + (1-alpha) * state.var
```

Test:

    tests/test_cuda_ewma_drift.py

Tolerance: dtype-abhängig (float64 strenger als float32)

### Stop Criterion

Keine zeitliche Drift.

------------------------------------------------------------------------

# STEP 5 --- Drawdown Port

Implementation:

``` python
state.peak_cum_pi = torch.maximum(state.peak_cum_pi, state.cum_pi)
state.drawdown = state.peak_cum_pi - state.cum_pi
```

Test:

    tests/test_cuda_drawdown_parity.py

### Stop Criterion

Peak-Verlauf identisch.

------------------------------------------------------------------------

# STEP 6 --- Rebirth Port

Implementation:

``` python
mask = state.wealth < state.rebirth_threshold
state.weights = torch.where(mask, reset_weight, state.weights)
```

Hinweis: `reset_weight` muss Shape \[B, N\] haben.

Test:

    tests/test_cuda_rebirth_parity.py

### Stop Criterion

Rebirth identisch.

------------------------------------------------------------------------

# STEP 7 --- 500-Step Full Simulation Parity

Test:

    tests/test_cuda_full_parity_500_steps.py

Seeds 0..4\
Profile A\
Regime Switch

Tolerance: dtype-abhängig (float64 strenger als float32)

### Stop Criterion

Alle Seeds grün.

------------------------------------------------------------------------

# STEP 8 --- CUDA Determinismus Test

Test:

    tests/test_cuda_determinism.py

Run: gleicher Seed zweimal → identical tensor outputs.

### Stop Criterion

Determinismus bestätigt.

------------------------------------------------------------------------

# STEP 9 --- Batch \> 1 Test

Test:

    tests/test_cuda_batch_consistency.py

Procedure: - CPU seed 0 - CPU seed 1 - CUDA batch B=2

Assertion: Batch\[0\] == CPU seed0\
Batch\[1\] == CPU seed1

### Stop Criterion

Batch verhält sich wie isolierte Runs.

------------------------------------------------------------------------

# STEP 10 --- Performance Benchmark (Optional)

Test:

    tests/test_cuda_benchmark.py

Vergleich CPU vs CUDA bei N \>= 10k.

Keine Assertion notwendig, nur Logging.

Hinweis: Optional. Nicht erforderlich für v0.6.4-alpha Abnahme.
Aktuelle RegimeSwitchBanditWorld ist K=5 fest verdrahtet; für echte N>=10k
braucht es eine separate Benchmark-World oder einen synthetischen r_vec-Generator.

------------------------------------------------------------------------

# Definition von v0.6.4-alpha DONE

Erfüllt wenn: - Alle Parity-Tests grün - Batch-Test grün - Kein Drift -
CPU-Referenz unverändert

------------------------------------------------------------------------

Ende -- Implementierungsplan Phase G G4
