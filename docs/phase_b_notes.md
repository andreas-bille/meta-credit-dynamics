# Phase B – Disembodied CapitalSelector
## Exploration, Aggregation, Kosten und strukturelle Selektion

Diese Notizen dokumentieren die konzeptionellen Überlegungen zu **Phase B**
des Modells *GAP – General Artificial Protagonist*.

Phase B beschreibt **keine Semantik, keine Planung und keine soziale Welt**.
Sie behandelt ausschließlich **lokale Selektion unter strukturellen Kosten**
in einer unbekannten Welt.

---

## 1. Grundannahmen

### 1.1 Disembodied Mind
Der CapitalSelector (CS):

- hat keinen Körper
- kennt keine Ziele
- besitzt keine Repräsentation der Welt
- operiert ausschließlich über lokale Rückkopplung

Er ist ein **Selektionsmechanismus**, kein Agent im klassischen Sinn.

---

### 1.2 Welt–Adapter–Prinzip

Der CS ist **weltagnostisch**.

- Die Welt wird ausschließlich über einen Adapter angebunden:
  ```
  (state, action) → (reward, cost, hit)
  ```
- Der CS bleibt **unverändert**, wenn die Welt wechselt
- Unterschiedliche Welten testen die **Struktur**, nicht den Code

---

## 2. Phase-B-Welt: Asymmetrische Kosten

### 2.1 Grundidee
Phase-B-Welten besitzen:

- lokale Beobachtbarkeit
- asymmetrische Konsequenzen
- keinen Zielgradienten

Fehler sind:
- leicht zu machen
- teuer zu korrigieren

---

### 2.2 Beispiel: MazeWorld

- Sichtbar: (x, y)
- Verborgen: z
- Aktionen: N, E, S, W
- Aufstieg: Δz ≤ +1
- Abstieg: Δz ∈ {0, −1, −2, …}
- Untere Schranke: z ≥ z_min

**Kernsatz:**

> Die Welt ist nach unten beschränkt, aber asymmetrisch:  
> Aufstieg ist lokal limitiert, Abstieg nur strukturell begrenzt.

---

## 3. Exploration ohne Planung

Exploration entsteht nicht durch Ziele oder Neugier,
sondern durch:
- unvollständige Information
- stochastische Wahl
- verzögerte Konsequenzen

Ein erfolgreicher CS:
- exploriert diffus
- vermeidet extreme Rückfälle
- bildet keine Karte
- „versteht“ nichts

---

## 4. Stack-Struktur (ohne Semantik)

### 4.1 Motivation
Stacks bündeln mehrere Selektoren:
- ohne Wissen über deren interne Struktur
- ohne explizite Koordination
- ohne Supervisor

Ein Stack ist **kein Portfolio**, sondern ein **Schicksalsverbund**.

---

### 4.2 Aggregation kostet

Aggregation ist **nicht kostenlos**.

Formell:
C_stack = Σ C_i + C_agg  mit  C_agg > 0

---

### 4.3 Gemeinsame Existenzschwelle

„Man lebt zusammen oder man stirbt zusammen.“

W_min(stack) = Σ W_min(i) + Δ_cohesion ,  Δ_cohesion > 0

Diese Zusatzschwelle ist der **Preis der Zusammengehörigkeit**.

---

## 5. Substack-Selektion

Ein Substack fällt aus dem Stack, wenn:

Beitrag_i < Aggregationskosten_i + Burndown_i

Dies ist eine **Bilanzgleichung**, kein Urteil.
Leere Stellen sind erlaubt.

---

## 6. Rebirth – strukturerhaltend

- Rebirth ist erlaubt
- Struktur bleibt erhalten
- Individuen sind austauschbar, Rollen nicht

> Die Struktur lebt weiter, nicht das Individuum.

---

## 7. CUDA & Messung

- CUDA lohnt sich für viele Episoden / Seeds
- Visualisierung ist optional
- Kernmodell arbeitet mit aggregierten Metriken

---

## 8. Phase-B-Grenze

Phase B behandelt keine:
- Sprache
- Bedeutung
- Wahrheit
- soziale Koordination

---

## 9. Ausblick

Phase C betrifft Bindung, Spur, Sediment, Zusammengehörigkeit –  
aber **erst**, wenn Phase B trägt.

> Lieber reden als blind laufen –  
> Worte sind billiger als Abstürze.
