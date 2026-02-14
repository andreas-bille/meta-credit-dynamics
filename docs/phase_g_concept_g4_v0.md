# Phase G – G4 CUDA & PyTorch Formalisierung

*(Target: v0.6.4-*)*

---

## Ziel

Vollständige Tensorisierung der Kernmechanik unter Erhalt der kanonischen Semantik.

**Wichtig:**

* Keine neue Ökonomie
* Kein neues Lernen
* Kein neues Signal
* Nur Rechenformalisierung + GPU-Pfad

---

# 1. Designprinzipien

## 1.1 Canonical First

Profile A bleibt Referenz.

CPU-Run = Ground Truth
CUDA-Run = numerisch äquivalente Projektion

Toleranz:

```
|cpu - cuda| < 1e-7  (float64)
```

---

## 1.2 No Semantic Drift Rule

CUDA darf:

* nur Rechenort ändern
* keine Reihenfolgeabhängigkeiten verändern
* keine implizite Normalisierung einführen

Insbesondere:

* Exponentiated Gradient identisch
* Rebirth-Mask identisch
* Drawdown-Logik identisch

---

## 1.3 Determinismus

Erforderlich:

```
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
```

Seeds:

* world seed
* torch seed
* numpy seed

Reproduzierbarkeit ist Pflicht für Phase G.

---

# 2. Tensorisierungs-Mapping

## 2.1 Zustandsvektoren

Alle Kernzustände werden als Tensoren geführt (canonical dtype, default: float64).

```
weights:  Tensor[B, N]
wealth:   Tensor[B, 1]
mean:     Tensor[B, 1]
var:      Tensor[B, 1]
peak:     Tensor[B, 1]
drawdown: Tensor[B, 1]
cum_pi:   Tensor[B, 1]
peak_cum_pi: Tensor[B, 1]
rebirth_threshold: Tensor[B, 1]
```

Hinweis: `wealth` bleibt ein globaler Scalar pro Selector; B=1 entspricht dem bisherigen Verhalten.

Device:

```
device = torch.device("cuda") or "cpu"
```

Alle Zustände müssen blockweise transferierbar sein.

---

## 2.2 Exponentiated Gradient (Reweight)

Mathematische Form:

$$w_{t+1,i} = (w_{t,i} * exp(eta * score_{t,i})) / Z_t$$
$$Z_t = sum_j w_{t,j} * exp(eta * score_{t,j})$$


CUDA-Form:

```
scores = ...
w = w * torch.exp(eta * scores)
w = w / w.sum(dim=1, keepdim=True)
```

Keine in-place-Normalisierung mit Seiteneffekten.
Numerische Stabilität (z.B. clamp, epsilon) muss exakt der CPU-Implementierung entsprechen.
Keine zusätzlichen Stabilisierungsschritte einführen.

---

## 2.3 EWMA-Statistik

Mittelwert:

$$m_{t+1} = alpha * r_t + (1-alpha) * m_t$$

Tensorisiert:

```
mean = alpha * returns + (1-alpha) * mean
```

Varianz:

$$v_{t+1} = alpha * (r_t - m_t)^2 + (1-alpha) * v_t$$

Tensorisiert:

```
var = alpha * (returns - mean_prev)**2 + (1-alpha) * var
```

Wichtig: mean_prev ist explizit der alte mean-Wert vor dem Update.
Der neue mean darf nicht für die Varianzberechnung verwendet werden,
sonst entsteht Drift gegenüber der CPU-Referenz.

---

## 2.4 Drawdown

Kanonische Definition (Phase F / math-v1.md):

$$
DD_t = peak\_cum\_pi - cum\_pi
$$

mit

$$
peak\_cum\_pi \leftarrow \\max(peak\_cum\_pi,\\ cum\_pi)
$$

G4 übernimmt exakt diese Definition.

CUDA (kanonisch):
```
peak_cum_pi = torch.maximum(peak_cum_pi, cum_pi)
drawdown = peak_cum_pi - cum_pi
```

Keine relative Division durch peak.  
Keine semantische Änderung erlaubt.

---

## 2.5 Rebirth-Mask

Maskenlogik ohne Schleifen:

```
mask = wealth < rebirth_threshold
weights = torch.where(mask, reset_weight, weights)
```

Keine semantische Änderung gegenüber CPU.
Hinweis: `reset_weight` muss shape‑kompatibel zu `weights` sein (z. B. [B, N]), um stilles Broadcasting zu vermeiden.

---

# 3. cuda_state.py – Zielarchitektur

## 3.1 DeviceState

```
@dataclass
class DeviceState:
    weights: Tensor
    wealth: Tensor
    mean: Tensor
    var: Tensor
    peak: Tensor
    drawdown: Tensor
    cum_pi: Tensor
    peak_cum_pi: Tensor
```

---

## 3.2 Projection API

```
state.to(device)
state.to_cpu()
```

Round-Trip-Test verpflichtend:

```
cpu_state -> cuda -> cpu -> compare
```

Batch‑Lift (CPU → CUDA):
- **Default (Var A):** `toCuda()` fügt eine Batch‑Dimension hinzu.  
  Beispiel: weights `[N] -> [1, N]`, wealth `[1] -> [1, 1]`.
- **Optional (Var B):** `toCuda(batch=B)` repliziert den CPU‑State B‑mal  
  (oder nimmt eine Liste von CPU‑States und stapelt sie).

## 3.3 Dtype-Policy

- Default: canonical dtype wird preserved.
- Kein implizites Casting (z.B. float64 -> float32).
- Casting nur explizit und dokumentiert.
- Test-Toleranzen abhängig vom dtype (float64 strenger als float32).

---

# 4. Compute Separation Layer

Struktur:

```
core/
    cpu_impl.py
    cuda_impl.py
```

Runtime-Auswahl:

```
if device.type == "cuda":
    backend = CudaCore()
else:
    backend = CpuCore()
```

Keine Device-Checks verteilt im Code.

---

# 5. Autograd-Regel

Kein Gradient-Learning in G4.

```
with torch.no_grad():
    step()
```

Begründung:

* verhindert unnötigen Graph-Aufbau
* reduziert Speicher
* stabilisiert Performance

---

# 6. Batchisierung

Batching wird umgesetzt, aber B darf 1 sein.
Shape-Notation ist durchgehend batched; B=1 entspricht dem bisherigen (unbatched) Fall.

Mehrere Welten parallel:

```
weights: Tensor[B, N]
```

Reduktionen über dim=1.

---

# 7. Testmatrix v0.6.4

## 7.1 Round Trip

CPU → CUDA → CPU

## 7.2 Single Step Equivalence

Ein Schritt CPU vs CUDA

## 7.3 500-Step World Equivalence

Regime Switch World
Seed 0..4

Maximale Abweichung protokollieren.

## 7.4 Adversarial Phase-Shift (G3.4.12)

Stresstest unter SNR-Reversal.

---

# 8. Performance-Erwartung

* N <= 100 → CPU ausreichend
* N >= 10k → GPU sinnvoll
* Batchisierung → massive Beschleunigung

---

# 9. Was G4 NICHT tun darf

❌ Kein neues Score-System
❌ Keine neue Normalisierung
❌ Keine neue Volatilitätsdefinition
❌ Kein Learning-Rate-Drift

G4 ist rein strukturelle Projektion.

---

# 10. Empfohlener Umsetzungsplan

1. Batch‑fähige Tensorisierung der Zustandsvektoren (alle Shapes [B, N] bzw. [B])
2. Reweight CUDA‑Port (B=1 parity)
3. Drawdown + Rebirth‑Port (B=1 parity)
4. Runtime Device Switch (CPU/GPU)
5. Test Suite erweitern:
   - B=1 parity
   - B>1 consistency

Tag:

```
v0.6.4-alpha
```

---

Ende – Phase G G4 Formalisierung v0.6.4
