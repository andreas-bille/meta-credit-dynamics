# Implementierungsstatus – Meta-Credit Dynamics (v0.4.0)

Diese Tabelle grenzt **konzeptionelle Spezifikation** und **aktuellen Implementierungsstand**
über die Phasen B–E sauber voneinander ab.

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
|  | Sparsity (emergent) | 🟡 | Implizit; sichtbar über Kollaps |
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
|  | Strukturerhaltender Rebirth | ✅ | Sediment bleibt erhalten |
| **Masken & Rollen** | Explizite Maske `m` | ⏳ | Noch nicht nötig ohne Hebel |
|  | Bank- vs. Unternehmer-Rolle | ⏳ | Phase-C-Übergang |
| **Stack-Ökonomie** | Aggregationskosten (Stack) | ⏳ | Definiert, nicht umgesetzt |
|  | Substack-Herausfallen | ⏳ | Diskutiert |
| **Kommunikation** | Kosten-/Masken-Spuren | ⏳ | Kandidat für Phase C |
|  | Skalierende „Sprache“ | ⏳ | Nicht notwendig für Phase D/E |
| **Reparatur & Meta-Kredit (Phase D)** | Fluss-Caps (Rate / Magnitude) | ✅ | Implementiert inkl. Simplex-Renorm |
|  | Lag Injection (FIFO / EMA) | ✅ | Zeitentkopplung ohne Informationszuwachs |
|  | Soft Bail-out (ε, Threshold, Cooldown) | ✅ | Zeitaufschub, keine Rettung |
|  | Isolation / Scheidung | ✅ | Node-Masken, TTL optional |
|  | RepairPolicy-Abstraktion | ✅ | Deterministisch, ON/OFF, reversibel |
|  | Brokerpfade (emergent) | ✅ | Tod → Sediment → Ausschluss wirksam |
|  | Meta-Kredit (Zeitkaufen) | 🟡 | Funktional, nicht als explizite Variable |
|  | Versicherungen / Derivate | 🚫 | Bewusst nicht modelliert |
| **Sediment & Emergenz (Phase E)** | Sedimented Mediation Paths (SMP) | ✅ | Strukturelle Residuen toter Brokerpfade |
|  | Sediment als DAG (append-only) | ✅ | JSONL, monotone IDs, keine Löschung |
|  | Sediment = passiv / nicht lesbar | ✅ | Wirkung nur über Formation-Filter |
|  | Phase-lokale Ausschlussregeln | ✅ | Identische Konfigurationen verboten |
|  | Pair-Forbids (Feature-Flag) | 🟡 | Optional, konservative Erweiterung |
|  | Stack-Dissolution → Sediment | ✅ | Hook im StackManager |
|  | Formation-Filter via Sediment | ✅ | Hard reject, kein Soft-Penalty |
|  | Rebirth erhält Sediment | ✅ | SedimentAwareRebirthPolicy |
|  | Curriculum-Phasen (Teacher) | 🟡 | API definiert, Beispiele vorhanden |
|  | Freeze-Fähigkeit (implizit) | 🟡 | Lernstopp konzeptionell möglich |
| **Bewusst ausgeschlossen** | Semantik / Bedeutung | 🚫 | Methodisch ausgeschlossen |
|  | Planung / RL | 🚫 | Kein Policy-Agent |
|  | Agenten-Ziele / Utility | 🚫 | Kein Nutzenoptimierer |

---

## Einordnung (aktualisiert)

- **Phase B**: kanonisches Grundmodell (Investition, Kosten, Rebalancing)
- **Phase C**: Kredit, Rollen, Hebel (konzeptionell vorbereitet)
- **Phase D**: Reparaturmechanismen als Meta-Kredit,
  forensisch instrumentiert, ohne Optimierung
- **Phase E**: irreversible Strukturentstehung durch Sediment,
  rebirth-fähig, curriculum-kompatibel

🟡-Einträge markieren **bewusste Vereinfachungen**.  
⏳-Einträge markieren **Erweiterungspunkte**, keine Defizite.  
🚫-Einträge sind **Design-Verbote**, keine TODOs.

> **Nach Phase E gilt:**  
> Das Modell ist vollständig genug, um  
> – an sich selbst zu scheitern,  
> – aus Scheitern irreversible Struktur zu bilden,  
> – und aus dieser Struktur gezielt neu zu wachsen.  
>
> Lernen ist nicht mehr reversibel,  
> aber weiterhin kontrollierbar.

---

*Stand: Phase E abgeschlossen (v0.4.0).*
