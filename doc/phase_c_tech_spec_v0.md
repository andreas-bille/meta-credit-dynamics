# Phase C – Kreditmärkte: Technische Spezifikation v0

Status: Entwurf implementierbar • Datum: 2026-01-30
Geltungsbereich: Explorer/ Broker/ Stack auf Basis des CapitalSelector-Kerns

Ziel: Präzisierung der in phase_c_concept.md beschriebenen Bewegung in implementierbare Schnittstellen, Metriken und Regeln. Semantikfrei, lokal, skaleninvariant.

---

## 1. Ontologie und Rollen

- Explorer (Channel): liefert bei Kapitalzufuhr einen verzögerten, volatilen Rückfluss r sowie Kosten c und ein Survival-Signal alive.
- Broker (Inhibitor): beobachtet ausschließlich Zeitreihen-Metriken und setzt Kreditlimits/ Frequenzen. Kein Koordinator, keine Optimierung.
- Stack: aggregierte Identität mehrerer Explorer, exportiert selbst einen Channel mit stabilem Cashflow.
- CapitalSelector: bleibt der stackbare Kern und stellt Allokationsgewichte w bereit.

Hinweis: Explorer, Stack und Broker sind operational definiert, keine neuen ontologischen Typen neben dem CapitalSelector-Kern.

---

## 2. Schnittstellen (typsicher, semantikfrei)

### 2.1 Explorer-Adapter (Channel)

- step(weight: float) -> tuple[r: float, c: float, alive: bool, dt: float]
  - weight: gegenwärtige Kapitalzufuhr
  - r: realisierter Rückfluss in derselben Zeiteinheit
  - c: zugeordnete Kosten (Betrieb/ Info/ Kapitalkosten)
  - alive: True, wenn Akteur weiterlieferfähig; False signalisiert Marktaustritt
  - dt: verstrichene Zeit seit letzter Rückmeldung (freie Einheit)

Kontrakt: keine Zustands-/Weltinformation außer obigem; kein Erfolg/Fehler-Label.

### 2.2 Broker

- observe(explorer_id, r, c, alive, dt)
- decide_limits() -> dict[explorer_id, CreditPolicy]
- allocate(I_total: float) -> dict[explorer_id, weight]

CreditPolicy:
- limit: float (maximale Kapitalrate pro Zeiteinheit)
- min_interval: float (Mindestabstand zwischen Finanzierungen)
- blocked: bool (harte Sperre)

### 2.3 Stack

- add(explorer_id)
- remove(explorer_id)
- step(weight: float) -> tuple[r: float, c: float, alive: bool, dt: float]
- state() -> Kennzahlen (mu, vol, drawdown, size)

Stack exportiert die Summe seiner Mitglieder als stabilen Channel. Mitglieder explorieren innerhalb des Stacks nicht mehr nach außen.

---

## 3. Messgrößen (Rolling/ EWMA)

Alle Metriken werden als gleitende Schätzer geführt, keine festen Fenster nötig.

Parameter (Defaults v0):
- EWMA beta_mu = 0.1 (Mittelwert)
- EWMA beta_var = 0.1 (Varianz)
- Tail-Risk via EWMA-CVaR alpha = 0.1 (unteres 10%-Quantil, approximiert)
- Korrelation via EWMA-Kovarianz → ρ_ij
- Persistenz via run-length EWMA (E[r>0])
- Δt: gleitender Mittelwert der zwischenliegenden Zeiten

Abgeleitete Kennzahlen pro Explorer i:
- mu_i, vol_i = sqrt(var_i)
- cvar_i (negativer Erwartungswert im unteren Alpha-Schwanz)
- dd_i: laufender Max-Drawdown der kumulierten r-c
- surv_i: EWMA-Überlebensrate

Zwischen Explorer i, j:
- rho_ij: EWMA-Korrelation der Rückflüsse r_i, r_j

---

## 4. Inhibitionsregeln (nur Schwellen, keine Optimierung)

Regeln wirken additiv/kompositorisch. Hysterese einsetzen, um Flattern zu vermeiden.

Parameter (Startwerte v0, empirisch zu kalibrieren):
- τ_var: Volatilitätsobergrenze relativ zu Erwartung, z. B. vol_i > k_var · |mu_i| mit k_var=3
- τ_cvar: Tail-Risk-Grenze, z. B. cvar_i < -γ_tail mit γ_tail=1·σ_ref
- τ_dd: Drawdown-Grenze, z. B. dd_i > 5·σ_ref
- τ_corr: Korrelation-Sperre, |rho_ij|>0.8 in Kombination mit schlechtem Tail-Risk
- τ_dt: Mindestabstand der Finanzierungen, min_interval = p95(Δt)

Maßnahmen:
- Reduce limit: limit_i ← max(0, limit_i · λ_down) bei Verletzung (λ_down=0.5)
- Block: blocked_i ← True bei starker Tail-/DD-Verletzung
- Cooldown: min_interval_i ← max(min_interval_i, τ_dt)
- De-correlation cap: simultane Limits für korrelierte Paare (i,j) kappen

Der Broker vergibt niemals positives Urteil; er entfernt nur Allokationsfreiheit.

---

## 5. Stack-Bildung und -Auflösung

Kandidatenmenge K ⊂ Explorer, die folgende Bedingungen über T_agg (EWMA) erfüllen:
- niedrige Varianz: vol_i < τ_var_stack
- robuste Rückflüsse: mu_i > τ_mu_stack und cvar_i > -τ_tail_stack
- geringe wechselseitige Korrelation: |rho_ij| < τ_corr_stack für i≠j
- Persistenz: surv_i > τ_surv_stack

Bildung:
- Stack := neue Entität; Mitglieder i∈K werden dem Stack zugeordnet
- Explorer im Stack werden extern unsichtbar; der Stack exportiert einen Channel
- Kostenaufschlag: c_stack = Σ c_i + C_agg (C_agg > 0)

Auflösung/ Austritt:
- Wenn Beitrag_i < (Anteil an C_agg + Burndown_i) → Entfernen aus Stack
- Wenn Stack-Stabilität verloren geht (Var/Tail verletzt) → Stack auflösen

Startwerte:
- τ_var_stack = 2·|mu|, τ_mu_stack = 0, τ_tail_stack = 1·σ_ref, τ_corr_stack = 0.3, τ_surv_stack = 0.6

---

## 6. Rebirth (hierarchische Emergenz)

- Rebirth-Schwelle pro Entität: wealth < τ_rebirth → Reset ausgewählter Zustände
- Broker-Rebirth: sobald obere Ebene genügend Varianz zeigt, entsteht neuer Broker darüber; implementiert als zusätzlicher Broker, der den Stack-Channel als Explorer betrachtet.

Defaults:
- τ_rebirth_explorer = 0
- τ_rebirth_stack = 0
- τ_rebirth_broker: nicht erforderlich; Broker ist funktional

---

## 7. Ereignisschleife (Pseudo-Code)

1. Für jeden Explorer i, dessen min_interval erfüllt und nicht blocked:
   - w_i ← CapitalSelector.allocate()[i]
   - (r_i, c_i, alive_i, dt_i) ← Explorer_i.step(w_i)
   - Broker.observe(i, r_i, c_i, alive_i, dt_i)
2. Broker.decide_limits() anwenden → Gewichte clippen
3. CapitalSelector.feedback_vector(r_vec, c_total)
4. Rebirth prüfen und ggf. durchführen
5. Stack-Kandidaten prüfen, bilden/auflösen
6. Hierarchische Rebirth prüfen (Stacks als Explorer für höheren Broker)

---

## 8. Abnahmekriterien v0

- A1: Bei synthetischem Explorer mit hoher Varianz/negativem Tail-Risk wird die Kapitalzuweisung innerhalb N Schritten < ε.
- A2: Für 2-3 schwach korrelierte, positive Explorer bildet sich binnen M Schritten ein stabiler Stack mit geringerer Volatilität als jedes Mitglied.
- A3: Korrelierte Explorer mit schlechtem Tail-Risk werden nicht gleichzeitig finanziert (De-correlation cap wirksam).
- A4: Alive=False führt zum sofortigen Kapitalentzug und Entfernen aus Stack.
- A5: Rebirth setzt Gewichte auf Simplex-Gleichverteilung zurück und hebt wealth auf die Schwelle an.

Richtwerte: N≈50–200, M≈500–2000 je nach Welt/Noise.

---

## 9. Mapping auf vorhandenen Code

- CapitalSelector: economy/capitalmarket/capitalselector/core.py
  - nutzt EWMAStats, reweight_fn; feedback_vector(r_vec, c) bereits vorhanden
- Reweighting: economy/capitalmarket/capitalselector/reweight.py
- Rebirth: economy/capitalmarket/capitalselector/rebirth.py

Erweiterungen v0:
- Neuer Broker-Typ: economy/capitalmarket/capitalselector/broker.py
  - hält Metriken je Explorer (mu, var, cvar, dd, rho, dt, surv)
  - implementiert observe/decide_limits/allocate
- Stack-Entität: economy/capitalmarket/capitalselector/stack.py
  - kapselt Explorer, exportiert Channel, berechnet C_agg
- Explorer-Adapter-Beispiele: economy/capitalmarket/capitalselector/channels.py
  - synthetische Channels mit konfigurierbarer Varianz/Delay/Tail
- Tests: economy/capitalmarket/capitalselector/tests_phase_c.py inkl. A1–A5

---

## 10. Parameter/Config

YAML/Dict-basierte Parametrisierung:
- ewma: beta_mu, beta_var, alpha_tail
- thresholds: τ_var, τ_cvar, τ_dd, τ_corr, τ_dt
- actions: λ_down, cooldown_factor, C_agg
- stack: τ_*-stack, min_size, max_size

---

## 11. Offene Punkte (gezielte Experimente)

- CVaR-EWMA: effiziente Approximation ohne Speicherung großer Fenster
- Korrelation nur in schlechten Regimes (tail-conditional ρ)
- Hysterese-Bänder, um Schwellenflattern zu minimieren
- Δt und Finanzierungsfrequenz: unterschiedliche Welten kalibrieren

Ende v0.
