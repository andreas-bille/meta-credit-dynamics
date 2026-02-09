# Delta-Vorschläge zu `math.md` (v1)

Ziel: Präzisierungen, die `math.md` **implementierbar** machen und Konflikte mit den Grundlagen-Dokumenten auflösen.  
Status: **Vorschlag** (noch keine Änderung an `math.md`).

---

## 0. Scope und Abgrenzung (neu)

**Vorschlag (Präzisierung):**
`math.md` beschreibt die **Meta‑Credit Dynamics** als abstrahierte, semantikfreie Schicht.  
ETF‑Baseline, Debt/Equity, Sparsity‑Masken und Broker‑Inhibition sind **außerhalb** des Scopes, sofern nicht explizit referenziert.

**Begründung:**
Vermeidet Widerspruch zum kanonischen Fundamentalinvestor‑Modell, das detailreicher ist.

---

## 1. Einheitliche Fluss-Definitionen (präzisieren)

**Ist (implizit):**
Stats und Stabilität benutzen $\Pi$; Update-Score $s_k(t)$ ist undefiniert.

**Vorschlag (Definitionen):**

Für Einheit $i$, Kanäle $k=1,\dots,n$, Zeit $t$:

1) **Brutto‑Return des Kanals**
$$
r_k(t) \in \mathbb{R}
$$

2) **Gewichtete Brutto‑Rückflüsse**
$$
R_i(t) := \sum_{k} w_{ik}(t)\, r_k(t)
$$

3) **Kosten (skalare Gesamtkosten)**
$$
C_i(t) := C_{info}(t) + C_{activity}(t) + C_{aggregation}(t)
$$

4) **Nettofluss**
$$
\Pi_i(t) := R_i(t) - C_i(t)
$$

5) **Kanal‑spezifischer Netto‑Beitrag (für Reweighting)**
$$
\pi_{ik}(t) := w_{ik}(t)\, r_k(t) - \frac{w_{ik}(t)}{\sum_j w_{ij}(t)}\, C_i(t)
$$

**Kommentar:**  
Damit ist klar, wie **Kosten** kanalweise aufgeteilt werden (proportional zum Gewicht), ohne zusätzliche Semantik.

---

## 2. Score $s_k(t)$ für Exponentiated Gradient (neu definieren)

**Vorschlag (kanonisch, minimal):**

Score pro Kanal:
$$
s_{ik}(t) := \pi_{ik}(t) - \mu_i(t)
$$

Update-Regel:
$$
w_{ik}(t+1) =
\frac{w_{ik}(t)\, \exp\big(\eta\, s_{ik}(t)\big)}
{\sum_j w_{ij}(t)\, \exp\big(\eta\, s_{ij}(t)\big)}
$$

**Begründung:**  
Die Statistik $\mu_i(t)$ ist skalar (Einheits‑Ebene) und erlaubt ein zentrales Nullniveau.  
Der Score basiert auf **Netto‑Beitrag** und ist damit konsistent mit der Definition von $\Pi_i$.

---

## 3. Statistik‑Definitionen (präzisieren)

**Vorschlag:**
Stats werden **einheitlich** auf $\Pi_i(t)$ definiert:

$$
\mu_i(t) = (1-\beta)\mu_i(t-1) + \beta \Pi_i(t)
$$
$$
\sigma_i^2(t) = (1-\beta)\sigma_i^2(t-1) + \beta\big(\Pi_i(t) - \mu_i(t)\big)^2
$$

Drawdown auf kumuliertem Nettofluss:
$$
DD_i(t) = \max_{\tau \le t}\left(\sum_{u=\tau}^{t} -\Pi_i(u)\right)
$$

---

## 4. Stabilitätsprädikat (Lücke schließen)

**Ist:** $S_j(t)$ unklar.

**Vorschlag (explizite Metriken):**

Definiere pro Einheit $i$ den Stabilitätsvektor:
$$
S_i(t) := \big(\mu_i(t),\ \sigma_i(t),\ DD_i(t)\big)
$$

Stabilität:
$$
\text{stable}_i(t) :=
\big(\mu_i(t) \ge \tau_\mu\big)\ \land\
\big(\sigma_i(t) \le \tau_\sigma\big)\ \land\
\big(DD_i(t) \le \tau_{dd}\big)
$$

**Kommentar:**  
Damit ist das Prädikat implementierbar und konsistent mit der Stack‑Stabilität.

---

## 5. Stack‑Aggregation (Gewichte definieren)

**Ist:** $\alpha_i$ undefiniert.

**Vorschlag (minimal, semantikfrei):**
Gleichgewichtet oder explizite Stack‑Weights:

1) **Gleichgewichtung:**
$$
\alpha_i = \frac{1}{|S|}
$$

2) **Allgemeine Definition:**
$$
\alpha_i \in \Delta^{|S|},\quad \sum_{i\in S}\alpha_i = 1
$$

Stack‑Return:
$$
R_S(t) = \sum_{i\in S} \alpha_i R_i(t)
$$

---

## 6. Sediment‑Node (fehlendes $w$ präzisieren)

**Ist:** $\nu = (M, P, w, \phi, t_d, r)$ ohne Bedeutung für $w$.

**Vorschlag (minimal):**
Interpretation von $w$ als **Stack‑Gewichtssignatur** zum Dissolve‑Zeitpunkt:
$$
w := \big(w_{i}(t_d)\big)_{i \in M}
$$

Falls nicht benötigt:
Alternative ist, $w$ aus der Definition zu entfernen.

---

## 7. Rebirth (State‑Reset explizit)

**Vorschlag (präzise Minimalvariante):**

Rebirth setzt folgende Zustände zurück:
$$
w_i(t+1) \leftarrow \text{uniform simplex}
$$
$$
\mu_i(t+1) \leftarrow 0,\quad \sigma_i^2(t+1)\leftarrow \sigma^2_{\text{seed}}
$$
$$
DD_i(t+1)\leftarrow 0
$$

Wealth-Reset:
$$
\text{wealth}_i(t+1) \leftarrow \max(\text{wealth}_i(t+1),\ \tau_{rebirth})
$$

**Kommentar:**  
Damit ist klar, was „state“ bedeutet.

---

## 8. Freeze (Inference Mode) präzisieren

**Vorschlag:**
Freeze setzt:
- **keine** neuen Einheiten/Stacks
- **keine** Sediment‑Insertion
- **kein** Rebirth
- **keine** Gewichts‑Updates (optional, je nach Ziel)

Formal:
$$
w_i(t+1) = w_i(t),\quad \mathcal{G}_{t+1}=\mathcal{G}_t,\quad S_{t+1}=S_t
$$
Stats dürfen weiter laufen (optional):
$$
\mu_i(t+1),\ \sigma_i^2(t+1)\ \text{update oder freeze? (explizit festlegen)}
$$

---

## 9. Konsistenzhinweis zu Grundlagen‑Dokumenten

Wenn `math.md` als **übergeordnete, abstrahierte Ebene** gilt:
- ETF‑Baseline, Alpha, Debt/Equity etc. sind **vorgelagerte Modelle** (Fundamentalinvestor‑Spezifikation).
- `math.md` bleibt eine **reduzierte, semantikfreie** Form.

Wenn `math.md` hingegen als **kanonische Implementations‑Spezifikation** gilt:
- Dann müssen ETF‑Baseline, Sparsity‑Maske, Kredit‑Inhibition etc. **eingebaut oder referenziert** werden.

---

## 10. Kurze Checkliste für spätere Umsetzung

- [ ] Nettofluss als einziges Statistik‑Signal verwenden  
- [ ] Score $s_{ik}$ formal definieren  
- [ ] Stabilitätsprädikat explizit machen  
- [ ] $\alpha_i$ in Stacks definieren  
- [ ] Rebirth‑State präzisieren  
- [ ] Freeze‑Semantik festlegen  
- [ ] Sediment‑Node $w$ definieren oder entfernen  

---

## 10.1 Konzept‑Hintergrund: Learning as Inhibition

**Ziel:** Festhalten, dass Lernen **nicht** als Optimierung, sondern als **Inhibition** beschrieben wird.

**Kernidee (prägnant):**
Das System lernt primär dadurch, dass **schlechte Pfade ausgeschlossen** werden.  
Es gibt **keine** positive Bewertung, keine Ziel‑Funktion, keine Semantik.  
„Lernen“ bedeutet hier: der Raum zulässiger Strukturen schrumpft **irreversibel**.

**Formale Lesart (kompatibel mit math.md):**
- Inhibition wirkt über **harte Ausschlussregeln** (z. B. Sediment‑Exclusion).
- Stabilität ist **Schwellen‑basiert** (keine Nutzenmaximierung).
- Der verbleibende Strukturraum ist das Ergebnis des **Ausschlusses**, nicht der Suche.

**Implikationen für Implementierung:**
- Policies sind **negativ** (defund / block / dissolve), nicht „rewarding“.
- Jede Erweiterung muss diese Asymmetrie explizit respektieren.

---

## 11. ETF‑Baseline und Overperformance (integrieren)

**Ist (fehlend in `math.md`):** Keine Referenz‑Baseline, keine relative Performance.

**Vorschlag (formale Ergänzung):**

Definiere eine Referenz‑Allokation $q(t) \in \Delta^n$ (ETF‑Baseline):
$$
q(t) = \big(q_1(t), \dots, q_n(t)\big),\quad \sum_k q_k(t)=1,\quad q_k(t)\ge 0
$$

ETF‑Return:
$$
R^{ETF}(t) := \sum_k q_k(t)\, r_k(t)
$$

Brutto‑Overperformance:
$$
\Delta_i^{gross}(t) := R_i(t) - R^{ETF}(t)
$$

Netto‑Overperformance:
$$
\Delta_i^{net}(t) := \Pi_i(t) - R^{ETF}(t)
$$

**Optional (kanonisch):** Score kann auf $\Delta_i^{net}$ basieren, statt auf $\Pi_i$.

---

## 12. Aktivitätsgrad und Mischportfolio (integrieren)

**Ist (fehlend):** Keine explizite aktive/ETF‑Mischung.

**Vorschlag (formale Ergänzung):**

Trenne aktives Portfolio $a_i(t)\in\Delta^n$ von Gesamtportfolio $w_i(t)$:
$$
w_i(t) = (1-\alpha_i(t))\, q(t) + \alpha_i(t)\, a_i(t)
$$

mit Aktivitätsgrad:
$$
\alpha_i(t) \in [0,1]
$$

Aktive Allokation wird per Exponentiated Gradient aktualisiert:
$$
a_{ik}(t+1) \propto a_{ik}(t)\, \exp\big(\eta\, g_{ik}(t)\big)
$$

**Minimaler Score:**
$$
g_{ik}(t) := r_k(t) - R^{ETF}(t)
$$

---

## 13. Kapitalstruktur (Equity/Debt) und Kosten (integrieren)

**Ist (fehlend):** Keine Equity/Debt‑Dynamik, keine Kapitalerhaltung.

**Vorschlag (formale Ergänzung):**

Zustand:
$$
E_i(t) \ge 0,\quad D_i(t) \ge 0,\quad K_i(t)=E_i(t)+D_i(t)
$$

Brutto‑Profit (auf gesamtes Kapital):
$$
\Pi_i^{gross}(t) := K_i(t)\, R_i(t)
$$

Kostenblöcke:
$$
\text{Cost}_i(t) := D_i(t)c_{D,i}(t) + E_i(t)c_{E,i}(t) + b_i(t)
$$

Netto‑Profit:
$$
\Pi_i^{net}(t) := \Pi_i^{gross}(t) - \text{Cost}_i(t)
$$

Equity‑Update:
$$
E_i(t+1) := E_i(t) + \Pi_i^{net}(t)
$$

**Kommentar:**  
Damit ist klar, wie $R_i(t)$ in Kapital- und Kosten‑Dynamik einfließt.

---

## 14. Kreditkonditionen (Inhibition) und Austrocknung (integrieren)

**Ist (fehlend):** Keine formale Kredit‑/Inhibitionsdynamik.

**Vorschlag (formale Ergänzung):**

Kreditkondition als Funktion von Risiko/Performance:
$$
\psi_i(t) := \sigma\big(a_i\,\mu_i(t) - b_i\,\sigma_i(t) - d_i\,DD_i(t)\big)
$$

Debt‑Update:
$$
D_i(t+1) = \operatorname{clip}_{[0,D_{\max,i}]}\Big(\ell_{\max,i}\,E_i(t+1)\,\psi_i(t)\Big)
$$

**Austrocknung (monoton bei Verschlechterung):**
$$
D_i(t+1) \le D_i(t)\quad \text{falls}\quad \psi_i(t+1) < \psi_i(t)
$$

---

## 15. Sparsity‑Maske (integrieren)

**Ist (fehlend):** Keine formale Masken‑Regel.

**Vorschlag (formale Ergänzung):**

Maske $m_i(t)\in\{0,1\}^n$:
$$
a_{ik}(t) = 0\quad \text{falls}\quad m_{ik}(t)=0
$$

Masken‑Update (minimal, deterministisch):
$$
m_{ik}(t+1)=
\begin{cases}
0 & \text{falls } a_{ik}(t)\approx 0 \text{ über } H \text{ Schritte} \\
1 & \text{falls } k \sim q(t) \text{ (sparsame Exploration)}
\end{cases}
$$

---

## 16. Rebirth‑Pool und Kapitalerhaltung (integrieren)

**Ist (fehlend):** Keine Kapitalerhaltungslogik über Exits hinweg.

**Vorschlag (formale Ergänzung):**

Rebirth‑Pool $P(t)$:
$$
P(t+1) \leftarrow P(t) + \max(0, K_i(t+1))
$$

Seed‑Kapital aus Pool:
$$
E_i(t+1) \leftarrow E_{seed},\quad P(t+1) \leftarrow P(t+1) - E_{seed}
$$

**Kommentar:**  
Damit ist „kapitalerhaltender Rebirth“ formal fixiert.

---

## 17. Implementierungs‑Statusmarker (für Architektur‑Mapping)

**Vorschlag (Markierung pro Abschnitt in `math.md`):**
Jeder neue Block erhält eine Status‑Zeile:
```
Status: [implemented | partial | planned]
```

Beispiel:
```
Status: planned (not in code as of 2026‑02‑08)
```

So kann die Architektur‑Doku präzise referenzieren, was fehlt.

---

## 18. Production‑Delta (Minimalanforderungen für erste produktive Schritte)

Ziel: **konkrete, testbare Lückenliste**, die von „metaphorisch“ → „implementierbar“ führt.

### 18.1 Harte Spezifikationen (ohne Metaphern)

- **Score-Definition**  
  **Fehlt:** präzise Definition von $s_{ik}(t)$ im Reweighting.  
  **Delta:** explizit festlegen, ob $s_{ik}$ auf $\pi_{ik}$, $\Pi_i$ oder $\Delta_i$ basiert; inkl. Kostenverteilung.

- **Stabilitätsprädikat**  
  **Fehlt:** Definition der $S_j(t)$ und Schwellen $\tau_j$.  
  **Delta:** fest definieren (z. B. $\mu,\sigma,DD$) inkl. Einheiten.

- **Rebirth‑State**  
  **Fehlt:** präzise Reset‑Menge.  
  **Delta:** formale Liste der Zustände, die resetten (weights, stats, flags, wealth?).

- **Freeze‑Semantik**  
  **Fehlt:** was bleibt konstant, was läuft weiter?  
  **Delta:** explizite Update‑Regeln für $w$, Stats, Sediment, Stacks.

### 18.2 Daten- und Zustandsmodell

- **Persistenter Zustand**  
  **Fehlt:** formaler State‑Schema pro Einheit/Stack/Broker.  
  **Delta:** definieren, welche Felder persistent sind und wie sie initialisiert werden.

- **Zeitdefinition**  
  **Fehlt:** Semantik von $t$ / $dt$ (diskret? variable Schrittweite?).  
  **Delta:** festlegen und in allen Formeln konsistent verwenden.

### 18.3 Minimaler produktionsnaher Loop

- **Engine‑Loop**  
  **Fehlt:** kanonische Schrittfolge.  
  **Delta:** fixieren: Observe → Update Stats → Policy/Weights → Apply → Feedback → Maintenance.

- **Determinismus**  
  **Fehlt:** definierte RNG‑Seeding‑Regel.  
  **Delta:** obligatorisch für Tests und Reproduzierbarkeit.

### 18.4 Akzeptanzkriterien (Test‑Invarianten)

- **Sediment‑Monotonie** (append‑only, niemals löschen)  
- **Rebirth‑Reset** (Stats/Weights exakt zurückgesetzt, wenn vorgesehen)  
- **Freeze‑Invarianten** (keine Topologie‑Änderungen, keine Sediment‑Insertion)  
- **Budget‑Invarianten** (Weights auf Simplex, keine negativen Gewichte)

### 18.5 Status-Mapping (Architektur ↔ Implementierung)

- **Pro Abschnitt** (math.md): `Status: implemented|partial|planned`.  
- **Pro Modul** (Code): Zuordnung, welche Formeln umgesetzt sind.  
- **Delta‑Tabelle**: Abschnitt ↔ Datei ↔ Status.

---

## 19. Minimaler Scope für „Prod v0“

Vorschlag für einen **kleinen, erreichbaren** Produktions‑Scope:

1) **Netto‑basiertes Reweighting** (Score + Stats + Costs konsistent)  
2) **Stabilität + Stack‑Dissolution** (mit klaren Schwellen)  
3) **Sediment‑Exclusion** (DAG append‑only, Phase‑lokal)  
4) **Rebirth‑Reset** (explizit, getestet)  
5) **Freeze‑Mode** (Topologie fix, optional Stats‑update)

Alles andere (ETF‑Baseline, Debt/Equity, Sparsity‑Masken, Kredit‑Austrocknung) kann in **Prod v1** kommen, aber muss im kanonischen Dokument bereits formalisiert sein.
