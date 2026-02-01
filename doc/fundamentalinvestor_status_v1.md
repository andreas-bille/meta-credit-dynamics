# Implementierungsstatus – Fundamentalinvestor‑Neuron (Canonical v1)

Diese Tabelle grenzt **konzeptionelle Spezifikation** und **aktuellen Implementierungsstand** sauber voneinander ab.
Sie ist bewusst **wertfrei**: „nicht umgesetzt“ bedeutet *offen*, nicht *falsch*.

---

## Legende
- ✅ = umgesetzt / funktional vorhanden  
- 🟡 = teilweise / vereinfacht umgesetzt  
- ⏳ = konzeptionell definiert, noch nicht implementiert  
- 🚫 = bewusst nicht Teil von v1

---

## Übersicht

| Bereich | Element | Status | Kommentar |
|------|--------|--------|-----------|
| **Grundmodell** | Weltagnostik (nur Renditen & Kosten) | ✅ | Keine Weltzustände, keine Aktionen |
|  | Stacking (Neuron investiert in Neuron) | ✅ | Rekursiv nutzbar |
|  | Long‑only + Simplex | ✅ | Gewichte normalisiert |
|  | Sparsity (emergent) | 🟡 | Implizit über Gewichtszusammenbruch |
| **ETF‑Baseline** | Referenz‑ETF `q` | 🟡 | Implizit oder konstant |
|  | Mischung `(1−α)q + αa` | ⏳ | α/ETF aktuell nicht implementiert |
| **Rendite & Statistik** | Bruttorendite `R_i` | ✅ | Direkt aus Gewichten |
|  | ETF‑Outperformance `Δ_i` | 🟡 | Derzeit Vorteil ggü. EMA (kein expliziter ETF) |
|  | Rolling Mean (EMA) | ✅ | Implementiert |
|  | Rolling Varianz | 🟡 | Ohne Kovarianzen |
| **Kosten** | Burndown / Informationskosten | ✅ | Expliziter Term |
|  | Aktivitätskosten (α‑abhängig) | 🟡 | Teilweise(indirekt), kein expliziter α-Term|
|  | Sparsity‑Kosten | ⏳ | Konzeptionell, noch nicht explizit |
| **Rebalancing** | Exponentiated Gradient | ✅ | Code‑nah umgesetzt |
|  | Asset‑Scores vs. ETF | 🟡 | Minimalform ggü. EMA, nicht ETF |
| **Leverage / Kredit** | Explizite Debt‑Variable `D_i` | ⏳ | Noch nicht modelliert |
|  | Kreditkondition `ψ_i` | ⏳ | Phase‑C‑relevant |
|  | Austrocknung / Margin Call | ⏳ | Nicht vorhanden |
| **Rebirth / Exit** | Exit‑Trigger (Equity < Schwelle) | 🟡 | Vereinfacht |
|  | Kapitalerhaltender Pool | ⏳ | Konzeptionell |
|  | Strukturerhaltender Rebirth | 🟡 | Reset ohne Pool |
| **Masken & Rollen** | Explizite Maske `m` | ⏳ | Noch nicht nötig ohne Hebel |
|  | Bank‑ vs. Unternehmer‑Rolle | ⏳ | Phase‑C‑Übergang |
| **Stack‑Ökonomie** | Aggregationskosten (Stack) | ⏳ | Definiert, nicht umgesetzt |
|  | Substack‑Herausfallen | ⏳ | Diskutiert |
| **Kommunikation** | Kosten‑/Masken‑Spuren | ⏳ | Kandidat für Phase C |
|  | Skalierende „Sprache“ | ⏳ | Noch rein konzeptionell |
| **Bewusst ausgeschlossen** | Semantik / Bedeutung | 🚫 | Methodisch ausgeschlossen |
|  | Planung / RL | 🚫 | Nicht Teil des Modells |
|  | Agenten‑Ziele | 🚫 | Kein Utility‑Agent |

---

## Einordnung

- **Phase B** ist vollständig lauffähig ohne rote Felder.
- 🟡‑Einträge sind **bewusste Vereinfachungen**, keine Lücken.
- ⏳‑Einträge markieren **echte Erweiterungspunkte** (Phase C / v2).
- 🚫‑Einträge sind **Design‑Verbote**, keine TODOs.

> **Das Modell ist in sich konsistent:  
> Implementierung ⊂ Spezifikation, nicht umgekehrt.**

---

*Stand: Diskussion nach Phase B, vor Phase C.*
