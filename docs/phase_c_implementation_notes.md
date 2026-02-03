
# Phase C – Implementierungsnotizen

Stand: 2026-01-30

Diese Notizen dokumentieren die konkret implementierten Bausteine aus
`doc/phase_c_tech_spec_v0.md` im Python-Modul `capitalmarket.capitalselector`.

## Module

### `capitalselector/broker.py`
- `PhaseCChannel`: Phase‑C Channel‑Interface (`step(weight)->(r,c,alive,dt)`)
- `LegacyChannelAdapter`: Adapter für bestehende Channels (`step->(r,c)`)
- `CreditPolicy`: (limit, min_interval, blocked)
- `BrokerConfig`: Default‑Parameter (EWMA, Schwellen, Actions)
- `Broker`: Inhibitor
  - `observe(...)` führt EWMA‑Metriken pro Explorer
  - `decide_limits()` berechnet Policies (nur Inhibition)
  - `apply_policies(weights)` wendet Sperre/Limit/Interval an + renormalisiert
  - `apply_decorrelation_cap(weights)` optionaler Korrelations‑Cap (sparsam)

**Metriken (v0):**
- EWMA‑Mittelwert/Varianz auf `r`
- Online‑Quantil (stochastic approximation) + EWMA‑CVaR‑Proxy auf Losses unter Quantil
- Max‑Drawdown auf Nettofluss `r-c`
- Survival‑EWMA
- dt‑EWMA
- Sparse EWMA‑Kovarianz/Korrelation (nur für aktive IDs)

### `capitalselector/stack.py`
- `StackChannel`: aggregierte Identität (exportiert selbst einen Phase‑C Channel)
  - gleichgewichtete interne Allokation
  - Kostenaufschlag `C_agg`
  - interne grobe Metriken (mu/vol/dd/cvar) auf Nettofluss
- `StackManager`: bildet/diversifiziert Stacks (v0) und löst instabile Stacks auf

**Hinweis:** v0 bildet „Diversifikations‑Stacks“ (niedrige Korrelation), weil sie
stabiler testen. Koordinations‑Stacks (hohe Korrelation bei gutem Tail‑Risk)
können in v1 ergänzt werden.

### `capitalselector/channels.py`
- `DummyChannel`: unverändert (nur Simplex‑Dimension K)
- synthetische Phase‑C Channels:
  - `GaussianExplorer`
  - `TailRiskExplorer`
  - `DeterministicExplorer`

### `capitalselector/tests_phase_c.py`
Unittest‑Suite mit Abnahmekriterien A1–A5.

## Ausführen

```bash
python -m capitalmarket.capitalselector.tests_phase_c
```

## Minimaler Event‑Loop (in Tests)
Der Event‑Loop ist bewusst minimal gehalten und entspricht der „Gating“‑Version:
1. `w_raw = selector.allocate()`
2. `broker.decide_limits()`
3. `w = broker.apply_policies(w_raw)` (+ optional decorrelation cap)
4. `channel.step(w_i)` nur für freigegebene IDs
5. `broker.observe(...)`, `selector.feedback_vector(...)`
6. `stack_mgr.maintain()` / `stack_mgr.try_form_stack()`

Damit ist Phase C in v0 als isolierbarer Baustein testbar.
