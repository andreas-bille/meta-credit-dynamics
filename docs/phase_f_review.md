# Phase F – Architektur- & Mathe‑Review

Datum: 2026‑02‑08

Ziel: Abgleich von Architektur (docs/architecture.md) und Mathematik (docs/math.md) mit der Implementierung und den Tests. Bewertung der Professionalität und konkrete Verbesserungsvorschläge inkl. offener Punkte.

---

## Executive Summary

- Tests: 15/15 grün via `make test-cpu` (containerisiert). Die verifizierten Eigenschaften decken Phase C/D/E ab und bestätigen die Kernaussagen aus den Dokus.
- Architektur/Mathematik stimmen großteils mit der Implementierung überein. Einige Formulierungen in den Dokus sind leicht unpräzise oder widersprechen kleineren Implementierungsdetails, aber ohne Funktionalität zu gefährden.
- Codequalität: solide, modular, deterministisch wo nötig, mit klaren Schnittstellen und einfacher Persistenz (JSONL‑Telemetry/Sediment). Verbesserungsfelder: Terminologie präzisieren (u. a. CVaR‑Proxy), Doku‑Konsistenz (Stacks), optionale „Freeze“‑Betriebsart, und Schärfung des Kanalscores `s_k` (brutto/netto, global/lokal).

---

## Abgleich Architektur ↔ Implementierung

- Kanäle: Trennung zwischen `Channel` (2‑Tupel r,c) und `PhaseCChannel` (4‑Tupel r,c,alive,dt) ist sauber umgesetzt.
  - Referenz: [capitalmarket/capitalselector/core.py](../capitalmarket/capitalselector/core.py), [capitalmarket/capitalselector/broker.py](../capitalmarket/capitalselector/broker.py), [capitalmarket/capitalselector/channels.py](../capitalmarket/capitalselector/channels.py)
- CapitalSelector: Simplex‑Gewichte mit Exponentiated‑Gradient Reweighting; Rebirth setzt Wealth ≥ Schwelle und Simplex auf Gleichverteilung.
  - Referenz: `CapitalSelector.feedback_vector()`/`rebirth()` in [capitalmarket/capitalselector/core.py](../capitalmarket/capitalselector/core.py)
- Broker (Inhibitor): rein hemmende Politik (Volatilität, Tail‑Risk, Drawdown, Cooldown/Min‑Interval) + optionale Dekorrelation‑Caps.
  - Referenz: `decide_limits()`, `apply_policies()`, `apply_decorrelation_cap()` in [capitalmarket/capitalselector/broker.py](../capitalmarket/capitalselector/broker.py)
- Stacks: Aggregation als `StackChannel` inkl. Aggregationskosten und Stabilitätsprädikat; `StackManager` formt/auflöst.
  - Referenz: [capitalmarket/capitalselector/stack.py](../capitalmarket/capitalselector/stack.py)
- Sediment: Append‑only DAG mit phasenlokalem Exklusionsfilter, persistiert als JSONL (Nodes + Kanten als Log‑Events).
  - Referenz: [capitalmarket/capitalselector/sediment.py](../capitalmarket/capitalselector/sediment.py)
- Phase‑D Repair: deterministische, reversible Policies (Isolation, Caps, Bailout, Lag) in fester Reihenfolge.
  - Referenz: [capitalmarket/capitalselector/repair.py](../capitalmarket/capitalselector/repair.py)

Bewertung: Architektur ist konsistent umgesetzt; Integrationspunkte (Broker ⇄ Selector ⇄ Channels ⇄ StackManager ⇄ Sediment/Telemetry) sind klar und einfach testbar.

---

## Abgleich Mathematik ↔ Implementierung

### Simplex & Reweighting (§1–2)

Mathe: $w_{ik}(t+1) = \dfrac{w_{ik}(t)\, e^{\eta s_k(t)}}{\sum_j w_{ij}(t)\, e^{\eta s_j(t)}}$.

Code: `exp_reweight(w, advantage, eta)` implementiert den Exponential‑Update und projiziert danach auf das Simplex.
- Referenz: [capitalmarket/capitalselector/reweight.py](../capitalmarket/capitalselector/reweight.py)

Anmerkung: In `CapitalSelector.feedback_vector()` wird `adv = r_vec − stats.mu` verwendet, wobei `stats.mu` ein skalarer EWMA des Gesamt‑r ist. Das ist zulässig (relative Vorteilsgewichtung gegen eine globale Basislinie), weicht jedoch von einem strikten kanallokalen Netto‑Score ab (siehe unten bei „Schärfungen“).

### Statistiken & Stabilität (§3)

- EWMA für Mittel/Varianz: umgesetzt mit `EWMAStats` (Mittel) und `EWMAStats` (Varianz‑Instanz; Feld `var` wird als Varianz genutzt). Drawdown als Max‑Drawdown über kumulierten Nettofluss ist vorhanden (`EWMADrawdown`).
  - Referenz: [capitalmarket/capitalselector/stats.py](../capitalmarket/capitalselector/stats.py), [capitalmarket/capitalselector/broker.py](../capitalmarket/capitalselector/broker.py)

Abweichung: Die Mathe zeigt die Formeln auf Netto $\Pi(t)$; im Code wird die EWMA von $r$ pro Explorer gepflegt, Drawdown jedoch korrekt auf Netto (r−c). Das ist konsistent, sofern die Doku explizit macht, dass $\mu,\sigma^2$ im Broker brutto sind und Stabilitätsmaße (z. B. DD) Netto nutzen.

### Stacks (§4)

Stack‑Rendite und ‑Kosten sind wie beschrieben implementiert. Export als neuer `PhaseCChannel` ist vorhanden.
- Referenz: [capitalmarket/capitalselector/stack.py](../capitalmarket/capitalselector/stack.py)

### Auflösung & Sediment (§5–6)

Irreversible Auflösung wird mittels Sediment‑DAG festgehalten; Reformation identischer Konfigurationen innerhalb derselben Phase wird unterbunden. Persistenz als JSONL ist „append‑only“.
- Referenz: [capitalmarket/capitalselector/stack.py](../capitalmarket/capitalselector/stack.py), [capitalmarket/capitalselector/sediment.py](../capitalmarket/capitalselector/sediment.py)

### Rebirth (§7)

Selector setzt Wealth/Weights zurück; Sediment bleibt extern erhalten. Hook über `RebirthPolicy` bzw. `SedimentAwareRebirthPolicy` vorhanden.
- Referenz: [capitalmarket/capitalselector/core.py](../capitalmarket/capitalselector/core.py), [capitalmarket/capitalselector/rebirth.py](../capitalmarket/capitalselector/rebirth.py)

### Freeze (§8)

In der Mathe als Inferenz‑Modus definiert. Ein expliziter „Freeze“‑Schalter ist in der aktuellen Implementierung nicht vorhanden, funktional aber leicht nachrüstbar.

---

## Abweichungen & begriffliche Unschärfen

1) Architektur‑Non‑Goal: In [docs/architecture.md](./architecture.md) steht „No explicit stack classes“, während `StackChannel`/`StackManager` vorhanden sind. Empfehlung: Doku anpassen (Stacks sind explizite strukturelle Identitäten).

2) Definition von $s_k(t)$: Die Mathe sagt „Score aus jüngsten Nettoflüssen“. Im Code ist `adv = r_k − μ_\mathrm{global}` (brutto r gegen globale Baseline). Zwei zulässige Varianten – bitte eine festlegen und dokumentieren:
   - Variante A (aktuell im Code): $s_k(t) := r_k(t) − \mu_\mathrm{global}(t)$; Kosten wirken separat über Wealth/Broker.
   - Variante B (kanallokal, netto): $s_k(t) := \Pi_k(t) − \hat{\mu}_k(t)$ mit $\Pi_k(t) = r_k(t) − c_k(t)$ und kanallokaler EWMA.

3) EWMA auf $\Pi$ vs. $r$: Die Formeln in §3 sind für Netto geschrieben. Code nutzt $r$ für Mittel/Varianz pro Explorer, DD jedoch auf Netto. Empfehlung: Doku präzisieren (Brutto‑Mittel/Varianz, Netto‑DD), oder Implementierung auf Netto umstellen, falls gewünscht.

4) CVaR‑Terminologie: `StackChannel._cvar` ist ein negativer EWMA‑Proxy von Nettoverlusten, kein striktes CVaR. In Doku und/oder Code (Docstring) als Proxy benennen, um Missverständnisse zu vermeiden.

5) Freeze‑Modus: In Mathe vorhanden, im Code noch nicht als Modus verdrahtet (Neuerzeugung/Sediment/Rebirth deaktivieren). Umsetzung wäre geradlinig.

6) Korrelationen & Zeitskalierung: `update_correlations()` nutzt „last_r − mu“ ohne dt‑Normalisierung. Für heterogene dt eine Skalierung erwägen oder dokumentieren, dass dt≈1 angenommen wird.

7) `CapitalSelector.stack_step()`: Erwartet 2‑Tupel‑Kanäle; Phase‑C‑Kanäle liefern 4‑Tupel. In Docstrings trennen bzw. `stack_step` als Legacy‑Pfad kennzeichnen.

8) `EWMAStats` für Varianz: Die Varianz‑Instanz pflegt intern auch ein µ‑Feld (ungenutzt). Dokumentieren oder separate Klasse für Klarheit erwägen.

---

## Professionalitätsbewertung

- Positiv: klare Modularität, Typisierung, deterministische Policies, einfache Persistenz (JSONL), gut zugeschnittene Tests, reproduzierbare Container‑Umgebung (Makefile, Dockerfile, CPU/GPU Profile).
- Verbesserungspotenzial: Doku‑Konsistenz (Stacks), Terminologie (CVaR‑Proxy), optionaler Freeze‑Modus, präzisere Spezifikation von `s_k(t)` und Brutto/Netto‑Unterscheidung in §3.

In Summe erfüllt die Implementierung professionelle Ansprüche; die genannten Präzisierungen erhöhen Nachvollziehbarkeit und Betriebssicherheit.

---

## Konkrete Empfehlungen (umsetzbar, low‑risk)

1) Doku‑Konsistenz
- [docs/architecture.md](./architecture.md): „No explicit stack classes“ entfernen/anpassen.
- [docs/math.md](./math.md):
  - $s_k(t)$ präzisieren (Variante A vs. B; Brutto/Netto, global/lokal).
  - Rebirth‑Resetumfang definieren (behalten `stats`? optionaler Reset via Policy?).
  - CVaR‑Proxy klar benennen.
  - Freeze‑Modus als Implementierungsziel markieren.

2) API/Dokstrings
- `exp_reweight(w, advantage, eta)`: Typenhinweis/Dokstring auf „np.ndarray | float“ für `advantage`.
- `EWMAStats` (Varianz‑Instanz): kommentieren, dass `var` die Varianz ist und `mu` ignoriert wird.
- `StackChannel.state()['cvar']`: in Docstring als „negativer Verlust‑EWMA (CVaR‑Proxy)“ kennzeichnen.

3) Betriebsmodi
- Freeze‑Flag in `Broker`, `StackManager`, `CapitalSelector`:
  - `Broker.decide_limits()`/`apply_policies()`: keine neuen Entscheidungen (Limits fix lassen).
  - `StackManager.try_form_stack()`/`maintain()`: keine Formation/Auflösung; Sediment deaktiviert.
  - `CapitalSelector.rebirth()`: deaktivieren.

4) Optional (abhängig von gewünschter Strenge)
- Variante B für `s_k(t)` umsetzen: per‑Kanal‑µ und optionale kanalweise Kosten $c_k$ führen, dann `adv = (r_k − c_k) − \hat{\mu}_k$.
- `update_correlations()` dt‑bewusst gestalten (z. B. normiert pro Schrittbreite).

---

## Offene Punkte zur Klärung

1) Soll $s_k(t)$ strikt Netto $\Pi_k$ reflektieren? Falls ja, wie werden kanalindividuelle Kosten bereitgestellt (aktuell nur `c_total` aggregiert)?

2) Rebirth: Sollen EWMA‑Statistiken zurückgesetzt werden? Falls nein, in §7 explizit festhalten („Stats persistieren über Rebirth“); falls ja, `RebirthPolicy`‑Option ergänzen.

3) Freeze: Ist ein harter Inferenzmodus produktrelevant (keine neuen Stacks/Sediment/ Rebirth)? Wenn ja, Scope definieren und Tests ergänzen.

4) Tail‑Risk‑Politik: Ist `hard_block_tail` standardmäßig erwünscht oder szenarioabhängig? Empfehlung: in Konfiguration/Docs hervorheben.

---

## Testzusammenfassung & Reproduzierbarkeit

- Phase C: Defund von High‑Risk, Stack‑Volatilitätsreduktion, Dekorrelation‑Cap, sofortiges Defund bei `alive=False`, Rebirth‑Simplex‑Reset.
- Phase D: Zeitgewinn ohne Optimierung (Isolation), Fragilitätsanstieg durch Bailout, Lag‑Signatur, Isolation‑Trade‑off, „Offers‑Fenster=0“‑Signatur.
- Phase E: Monotones Sediment‑Wachstum, Unterdrückung identischer Reformationen innerhalb der Phase, Canalisation‑Signatur.

Ausführung (Container, ohne lokale Python‑Installation):

```bash
make test-cpu
# optional, erfordert NVIDIA Container Toolkit
make test-gpu
```

Letztstand: 15 passed in ~0.24s (CPU‑Container).

---

## Anhänge: Kernreferenzen

- Selector/Channel: [capitalmarket/capitalselector/core.py](../capitalmarket/capitalselector/core.py)
- Reweighting: [capitalmarket/capitalselector/reweight.py](../capitalmarket/capitalselector/reweight.py)
- Broker/Inhibition: [capitalmarket/capitalselector/broker.py](../capitalmarket/capitalselector/broker.py)
- Stacks/Manager: [capitalmarket/capitalselector/stack.py](../capitalmarket/capitalselector/stack.py)
- Sediment (DAG): [capitalmarket/capitalselector/sediment.py](../capitalmarket/capitalselector/sediment.py)
- Repair (Phase D): [capitalmarket/capitalselector/repair.py](../capitalmarket/capitalselector/repair.py)
- Telemetry: [capitalmarket/capitalselector/telemetry.py](../capitalmarket/capitalselector/telemetry.py)
