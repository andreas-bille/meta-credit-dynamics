# Implementierungsstatus – Meta-Credit Dynamics (v0.3.1)

Diese Tabelle grenzt **konzeptionelle Spezifikation** und **aktuellen Implementierungsstand**
über die Phasen B–D sauber voneinander ab.

Sie ist bewusst **wertfrei**:
„nicht umgesetzt“ bedeutet *offen*, nicht *falsch*.

---

## Legende
- ✅ = umgesetzt / funktional vorhanden
- 🟡 = teilweise / vereinfacht umgesetzt
- ⏳ = konzeptionell definiert, noch nicht implementiert
- 🚫 = bewusst nicht Teil des Modells

---

## Übersicht

| Bereich | Element | Status | Kommentar |
|------|--------|--------|-----------|
| **Grundmodell** | Weltagnostik (nur Renditen & Kosten) | ✅ | Keine Weltzustände, keine Aktionen |
|  | Stacking (Neuron investiert in Neuron) | ✅ | Rekursiv nutzbar |
|  | Long-only + Simplex | ✅ | Gewichte normalisiert |
|  | Sparsity (emergent) | 🟡 | Implizit; in Phase D über Kollaps sichtbar |
| **ETF-Baseline** | Referenz-ETF `q` | 🟡 | Implizit oder konstant |
|  | Mischung `(1−α)q + αa` | ⏳ | α/ETF aktuell nicht implementiert |
| **Rendite & Statistik** | Bruttorendite `R_i` | ✅ | Direkt aus Gewichten |
|  | ETF-Outperformance `Δ_i` | 🟡 | Vorteil ggü. EMA, kein expliziter ETF |
|  | Rolling Mean (EMA) | ✅ | Implementiert |
|  | Rolling Varianz | 🟡 | Ohne Kovarianzen |
| **Kosten** | Burndown / Informationskosten | ✅ | Expliziter Term |
|  | Aktivitätskosten (α-abhängig) | 🟡 | Indirekt, kein expliziter α-Term |
|  | Sparsity-Kosten | ⏳ | Konzeptionell |
| **Rebalancing** | Exponentiated Gradient | ✅ | Code-nah umgesetzt |
|  | Asset-Scores vs. ETF | 🟡 | Minimalform ggü. EMA |
| **Leverage / Kredit** | Explizite Debt-Variable `D_i` | ⏳ | Nicht modelliert |
|  | Kreditkondition `ψ_i` | ⏳ | Phase-C-relevant |
|  | Austrocknung / Margin Call | ⏳ | Nicht vorhanden |
| **Rebirth / Exit** | Exit-Trigger (Equity < Schwelle) | 🟡 | Vereinfacht |
|  | Kapitalerhaltender Pool | ⏳ | Konzeptionell |
|  | Strukturerhaltender Rebirth | 🟡 | Reset ohne Pool |
| **Masken & Rollen** | Explizite Maske `m` | ⏳ | Noch nicht nötig ohne Hebel |
|  | Bank- vs. Unternehmer-Rolle | ⏳ | Phase-C-Übergang |
| **Stack-Ökonomie** | Aggregationskosten (Stack) | ⏳ | Definiert, nicht umgesetzt |
|  | Substack-Herausfallen | ⏳ | Diskutiert |
| **Kommunikation** | Kosten-/Masken-Spuren | ⏳ | Kandidat für Phase C |
|  | Skalierende „Sprache“ | ⏳ | Nicht notwendig für Phase D |
| **Reparatur & Meta-Kredit (Phase D)** | Fluss-Caps (Rate / Magnitude) | ✅ | Implementiert inkl. Simplex-Renorm |
|  | Lag Injection (FIFO / EMA) | ✅ | Zeitentkopplung ohne Informationszuwachs |
|  | Soft Bail-out (ε, Threshold, Cooldown) | ✅ | Zeitaufschub, keine Rettung |
|  | Isolation / Scheidung | ✅ | Node-Masken, TTL optional |
|  | RepairPolicy-Abstraktion | ✅ | Deterministisch, ON/OFF, reversibel |
|  | Brokerpfade (emergent) | 🟡 | Sichtbar über Telemetrie |
|  | Meta-Kredit (Zeitkaufen) | 🟡 | Funktional, nicht als Variable |
|  | Versicherungen / Derivate | 🚫 | Bewusst nicht modelliert |
| **Bewusst ausgeschlossen** | Semantik / Bedeutung | 🚫 | Methodisch ausgeschlossen |
|  | Planung / RL | 🚫 | Kein Policy-Agent |
|  | Agenten-Ziele / Utility | 🚫 | Kein Nutzenoptimierer |

---

## Einordnung (aktualisiert)

- **Phase B**: kanonisches Grundmodell (Investition, Kosten, Rebalancing)
- **Phase C**: Kredit, Rollen, Hebel (konzeptionell vorbereitet)
- **Phase D**: abgeschlossen – Reparaturmechanismen als Meta-Kredit,
  forensisch instrumentiert, ohne Optimierung

🟡-Einträge markieren **bewusste Vereinfachungen**.  
⏳-Einträge markieren **Erweiterungspunkte**, keine Defizite.  
🚫-Einträge sind **Design-Verbote**, keine TODOs.

> **Nach Phase D gilt:**  
> Das Modell ist vollständig genug, um an sich selbst zu scheitern –  
> und genau das ist die Erkenntnis.

---

*Stand: Phase D abgeschlossen (v0.3.1).*
