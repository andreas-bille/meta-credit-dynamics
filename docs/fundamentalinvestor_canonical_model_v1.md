# Fundamentalinvestor‑Neuron – kanonisches Modell (v1)

> Ziel: Ein **weltagnostisches, stackbares, sparsity‑fähiges** Neuronmodell, das *ausschließlich* über Kapitalflüsse, Rückflüsse und Portfolio‑Rebalancing „lernt“ (kein RL, kein Planning).

---

## 0. Notation & Konventionen

Zeit diskret: $t \in \mathbb{N}$ (z. B. Tage/Wochen).  
Es gibt $N$ Fundamentalinvestor‑Neuronen $i \in \{1,\dots,N\}$.

### Vektoren / Matrizen
- $\mathbf{w}_i(t) \in \mathbb{R}^N_{\ge 0}$: Portfolio‑Gewichte des Neurons $i$ auf andere Neuronen (Long‑only, sparsity‑fähig).
- $\mathbf{r}(t) \in \mathbb{R}^N$: Realisierte Renditen der Neuronen als „Assets“ in Periode $t$.
- $\mathbf{1}$: Einsvektor.

### Skalare pro Neuron $i$
- $E_i(t)$: Eigenkapital (Net Asset Value / Equity).
- $D_i(t)$: Fremdkapital (Debt).
- $K_i(t) := E_i(t) + D_i(t)$: Investierbares Gesamtkapital (AUM).
- $\ell_i(t) := D_i(t)/E_i(t)$: Leverage (für $E_i(t)>0$).
- $c_{D,i}(t)$: Schuldzins/Kapitalkosten auf Debt (pro Periode).
- $c_{E,i}(t)$: Eigenkapitalkosten/„Dividendenanspruch“ (pro Periode; optional).
- $b_i(t)$: Burndown-/Informationskosten (pro Periode).
- $s_i(t)$: „Skill“‑Parameter (optional; exogen oder endogen) zur Kosten‑/Risiko‑Modulation.

---

## 1. Zustandsvariablen eines Neurons

Der Zustand $x_i(t)$ eines Neurons ist:

$$
x_i(t) = \Big(E_i(t),\, D_i(t),\, \mathbf{w}_i(t),\, \mu_i(t),\, \sigma_i^2(t),\, \rho_i(t),\, \text{sparsity‑mask } \mathbf{m}_i(t)\Big)
$$

mit:
- **Kapitalzustand:** $E_i, D_i, K_i$
- **Portfoliozustand:** $\mathbf{w}_i$ (Long‑only, Summe $=1$ auf aktiven Support)
- **Rolling‑Statistik:** $\mu_i(t)$ (rolling mean der aktiven Overperformance), $\sigma_i^2(t)$ (rolling Varianz), $\rho_i(t)$ (rolling Risikoindikator / z. B. Drawdown‑Proxy)
- **Sparsity‑Maske:** $\mathbf{m}_i(t)\in\{0,1\}^N$ (zulässige Kanten; optional dynamisch)

### Portfolio‑Constraints
Long‑only + Budget:
$$
w_{ij}(t)\ge 0,\quad \sum_{j=1}^N w_{ij}(t)=1
$$
Sparsity:
$$
w_{ij}(t)=0 \text{ falls } m_{ij}(t)=0
$$

---

## 2. ETF‑Baseline + aktive Komponente

Wir definieren einen **ETF** als Referenzportfolio $\mathbf{q}(t)\in\mathbb{R}^N_{\ge 0}$ mit $\sum_j q_j(t)=1$.  
Typisch:
- **Gleichgewichts‑ETF:** $\mathbf{q}(t)=\frac{1}{N}\mathbf{1}$ (equal‑weight), oder
- **Kapitalgewichteter ETF:** $q_j(t)=\frac{K_j(t)}{\sum_k K_k(t)}$.

Das Portfolio des Neurons $i$ wird als Mischung modelliert:

$$
\mathbf{w}_i(t) = (1-\alpha_i(t))\,\mathbf{q}(t) \;+\; \alpha_i(t)\,\mathbf{a}_i(t)
$$

mit:
- $\alpha_i(t)\in[0,1]$: **Aktivitätsgrad** (0 = reiner ETF, 1 = vollständig aktiv).
- $\mathbf{a}_i(t)$: aktive Allokation (ebenfalls long‑only & budgetiert; sparsity‑maskiert).

Constraints für $\mathbf{a}_i$:
$$
a_{ij}(t)\ge 0,\quad \sum_j a_{ij}(t)=1,\quad a_{ij}(t)=0 \text{ falls } m_{ij}(t)=0
$$

**Interpretation:** Weltagnostische „Fundamental“-Investoren vergleichen sich stets gegen $\mathbf{q}$ und verschieben nur den Anteil $\alpha$ aktiv.

---

## 3. Rückfluss / Return / Varianz

### 3.1 Realisierte Renditen als Asset‑Returns
Jedes Neuron $j$ erzeugt in Periode $t$ eine **Asset‑Rendite** $r_j(t)$.  
Diese Rendite ist *endogen* durch dessen eigene Kapitaldynamik (siehe Abschnitt 4) — aber für das Portfolio‑Update von $i$ genügt:

$$
R_i^{\text{gross}}(t) = \sum_{j=1}^N w_{ij}(t)\, r_j(t) \;=\; \mathbf{w}_i(t)^\top \mathbf{r}(t)
$$

ETF‑Rendite:
$$
R^{\text{ETF}}(t) = \mathbf{q}(t)^\top \mathbf{r}(t)
$$

### 3.2 Overperformance vs. ETF
Aktive Überrendite („Alpha“ realisiert):
$$
\Delta_i(t) := R_i^{\text{gross}}(t) - R^{\text{ETF}}(t)
$$

Optional „Netto‑Alpha“ nach Kosten (siehe 5):
$$
\Delta_i^{\text{net}}(t) := \big(R_i^{\text{gross}}(t)-\text{Kosten}_i(t)\big) - R^{\text{ETF}}(t)
$$

### 3.3 Varianz / Risiko des Portfolios
Ein kanonisches, weltagnostisches Risiko‑Signal ist die **rolling Varianz** der Überrendite:

$$
\sigma_i^2(t) \approx \operatorname{Var}\big(\Delta_i(\tau)\big)_{\tau \le t}
$$

Alternativ/ergänzend: Varianz des Bruttoreturns $R_i^{\text{gross}}$.

---

## 4. Kapitaldynamik (Eigenkapital + Fremdkapital)

### 4.1 Ergebnis vor Kosten
„Asset‑Return“ wirkt auf das gesamte investierte Kapital $K_i(t)$:

$$
\Pi_i^{\text{gross}}(t) := K_i(t)\,R_i^{\text{gross}}(t)
$$

### 4.2 Kostenblöcke
Pro Periode definieren wir additive Kosten (in Geldeinheiten):

1) **Debt‑Kosten (Zinsen):**
$$
\text{Int}_i(t) := D_i(t)\,c_{D,i}(t)
$$

2) **Equity‑Kosten (Dividendenanspruch / Ausschüttung):** (optional, kann 0 sein)
$$
\text{Div}_i(t) := E_i(t)\,c_{E,i}(t)
$$

3) **Burndown / Informationskosten:** (moduliert durch Aktivität/Komplexität)
$$
b_i(t) := b_{0,i} + b_{\alpha,i}\,\alpha_i(t) + b_{\text{deg},i}\,\|\mathbf{m}_i(t)\|_0
$$
wobei $\|\cdot\|_0$ die Anzahl aktiver Kanten misst (Sparsity‑Kosten).

Gesamtkosten:
$$
\text{Cost}_i(t) := \text{Int}_i(t) + \text{Div}_i(t) + b_i(t)
$$

### 4.3 Netto‑Gewinn und Equity‑Update
Netto‑Profit:
$$
\Pi_i^{\text{net}}(t) := \Pi_i^{\text{gross}}(t) - \text{Cost}_i(t)
$$

Equity‑Update (kapitalerhaltend, Debt zunächst konstant; Debt‑Update siehe 7):
$$
E_i(t+1) := E_i(t) + \Pi_i^{\text{net}}(t)
$$

AUM für nächste Periode:
$$
K_i(t+1) := E_i(t+1) + D_i(t+1)
$$

> Hinweis: Negative Equity ist in v1 erlaubt *bis zur Exit‑Regel* (Abschnitt 8). Alternativ kann man bei $E_i(t+1)\le 0$ sofort „Exit“ triggern.

---

## 5. Relative Overperformance vs. ETF (netto, risiko‑bereinigt)

Wir definieren eine kanonische Zielgröße als **Sharpe‑ähnliches** Signal auf Überrendite:

$$
J_i(t) := \frac{\mu_i(t)}{\sqrt{\sigma_i^2(t)}+\varepsilon} \;-\; \lambda_i\,\alpha_i(t) \;-\; \kappa_i\,\|\mathbf{m}_i(t)\|_0
$$

mit:
- $\mu_i(t)$: rolling mean der $\Delta_i^{\text{net}}(t)$ oder $\Delta_i(t)$
- $\sigma_i^2(t)$: rolling var
- $\varepsilon>0$: numerische Stabilisierung
- $\lambda_i$: „Aktivitäts‑Penalty“ (Handels-/Modellrisiko)
- $\kappa_i$: Sparsity‑Penalty (Informations-/Aufmerksamkeitsbudget)

Dieses $J_i$ ist die **lokale Optimierungsgröße** für Rebalancing (Abschnitt 6) — ohne RL‑Reward, nur aus Kapital‑ und Statistikgrößen.

---

## 6. Rolling‑Statistik (Mean / Var / Delay implizit)

Nutze exponentielle Glättung (EMA/EWMA).  
Für ein generisches Signal $z_i(t)$ (z. B. $\Delta_i^{\text{net}}(t)$):

$$
\mu_i(t+1) = (1-\beta)\,\mu_i(t) + \beta\,z_i(t)
$$

EWMA‑Varianz:
$$
v_i(t+1) = (1-\beta)\,v_i(t) + \beta\,(z_i(t)-\mu_i(t))^2
$$
und setze $\sigma_i^2(t):=v_i(t)$.

**Delay implizit:** Kleine $\beta$ → langes Gedächtnis, große $\beta$ → kurzes Gedächtnis.  
Optional: unterschiedliche $\beta_\mu,\beta_\sigma$.

---

## 7. Portfolio‑Rebalancing (ohne RL)

### 7.1 Grundprinzip
Das Neuron verschiebt Gewicht in Richtung Assets $j$, die relativ zum ETF besser performen (und/oder günstigeres Risiko liefern), unter Sparsity‑Constraints.

Kanonisches *code‑nahes* Update: **Exponentiated Gradient** (mirror descent) auf Scores $g_{ij}(t)$.

Definiere per Asset $j$ einen Score:
$$
g_{ij}(t) := \widehat{\mu}_{ij}(t) - \gamma_i\,\widehat{\sigma}_{ij}(t) - \tau_i
$$
wobei $\widehat{\mu}_{ij}$ eine rolling Schätzung des relativen Beitrags von $j$ ist, und $\widehat{\sigma}_{ij}$ ein Risiko‑Proxy. In v1 genügt als Minimalform:
$$
g_{ij}(t) := r_j(t) - R^{\text{ETF}}(t)
$$
(„Outperformance‑Score“ gegen ETF).

Dann für die aktive Allokation:
$$
a_{ij}(t+1) \propto a_{ij}(t)\,\exp(\eta_i\, g_{ij}(t))
$$
mit Projektion auf Simplex + Maske:
- Setze $a_{ij}(t+1)=0$ falls $m_{ij}(t)=0$
- Normalisiere $\sum_j a_{ij}(t+1)=1$

### 7.2 Aktivitätsgrad $\alpha_i$
$\alpha_i(t)$ kann selbst dynamisch werden (konservativ vs. unternehmerisch):

$$
\alpha_i(t+1) = \operatorname{clip}_{[0,1]}\Big(\alpha_i(t) + \eta_\alpha \cdot \text{sign}(\mu_i(t)) - \eta_\sigma \cdot \sigma_i(t)\Big)
$$

Konservativ: kleine $\eta_\alpha$, hohe $\eta_\sigma$.  
Unternehmerisch: größere $\eta_\alpha$, niedrigere $\eta_\sigma$.

---

## 8. Kreditaufnahme- und Austrocknungsmechanismus

### 8.1 Debt‑Politik (einfach, kanonisch)
Debt ist eine Funktion von Equity und Risiko:

$$
D_i(t+1) = \operatorname{clip}_{[0,\,D_{\max,i}]}\Big(\ell_{\max,i}\,E_i(t+1)\cdot \psi_i(t)\Big)
$$

mit einer „Kreditkonditions“-Funktion $\psi_i(t)\in[0,1]$, die bei schlechter Performance/hohem Risiko austrocknet:

$$
\psi_i(t) := \sigma\big( a_i\,\mu_i(t) - b_i\,\sigma_i(t) - d_i\,\text{DD}_i(t)\big)
$$

- $\sigma(\cdot)$: Sigmoid.
- $\text{DD}_i(t)$: Drawdown‑Proxy (optional, kann $0$ sein oder aus Equity‑Historie geschätzt werden).

Interpretation:
- Gute, stabile Outperformance → $\psi\to 1$ → Leverage verfügbar.
- Hohe Varianz / Drawdown → $\psi\to 0$ → Kredit trocknet aus.

### 8.2 Margin Call / Deleveraging (Austrocknung)
Wenn $\psi_i(t)$ fällt, wird Debt reduziert (Refinanzierungslimit):

$$
D_i(t+1) \le D_i(t) \quad \text{falls}\quad \psi_i(t+1) < \psi_i(t)
$$

Eine konkrete Regel:
$$
D_i(t+1) := \min\big(D_i(t),\; \ell_{\max,i}\,E_i(t+1)\,\psi_i(t+1)\big)
$$

Damit entsteht ein **endogener Crash‑Mechanismus** ohne RL: Volatilität → Kreditentzug → geringere AUM → geringere Marktmacht.

---

## 9. Rebirth / Exit‑Regel (kapitalerhaltend)

Ziel: Neuron kann „sterben“ (Exit) und **kapitalerhaltend** neu starten (Rebirth), ohne Gesamtmasse zu zerstören.

### 9.1 Exit‑Trigger
Exit wenn:
$$
E_i(t+1) < E_{\min} \quad \text{oder}\quad K_i(t+1) < K_{\min}
$$

### 9.2 Kapitalerhaltender Rebirth
Beim Exit wird das verbleibende Kapital (typisch $K_i(t+1)\ge 0$) **in den ETF‑Pool** zurückgeführt und das Neuron neu initialisiert.

Kanonisch:
1) **Rückführung:**
$$
\text{Pool}(t+1) \mathrel{+}= \max(0, K_i(t+1))
$$
2) **Reset:**
$$
E_i(t+1)\leftarrow E_{\text{seed}},\quad D_i(t+1)\leftarrow 0,\quad \alpha_i(t+1)\leftarrow \alpha_{\text{seed}}
$$
3) **ETF‑Startportfolio:**
$$
\mathbf{w}_i(t+1)\leftarrow \mathbf{q}(t+1)
$$
4) **Statistik‑Reset:**
$$
\mu_i(t+1)\leftarrow 0,\quad \sigma_i^2(t+1)\leftarrow \sigma_{\text{seed}}^2
$$

**Kapitalerhaltend** heißt hier: Das System hat einen Pool/Reserve‑Account, der Exit‑Kapital sammelt und (optional) seed‑Kapital bereitstellt.  
Minimal kann $E_{\text{seed}}$ aus dem Pool entnommen werden:
$$
\text{Pool}(t+1) \mathrel{-}= E_{\text{seed}}
$$
sofern Pool ausreichend gedeckt ist (sonst kein Rebirth oder kleineres Seed).

---

## 10. Hinweise zu Sparsity

Sparsity ist ein First‑Class‑Constraint:
- $\mathbf{m}_i(t)$ begrenzt die Anzahl der beobachteten/handelbaren Gegenparteien.
- Kosten $b_i(t)$ steigen mit $\|\mathbf{m}_i(t)\|_0$.
- Die aktive Allokation $\mathbf{a}_i$ wird nur über maskierte Kanten aktualisiert.

### 10.1 Kanonische Masken‑Regeln (ohne RL)
Masken‑Update kann rein aus Kapitalflüssen entstehen:

- **Pruning:** Entferne Kante $j$ wenn über $H$ Perioden:
$$
w_{ij} \approx 0 \quad \text{und/oder}\quad \widehat{\mu}_{ij}<\theta_{\text{drop}}
$$
- **Exploration (sparsam):** Erlaube periodisch $k$ neue Kanten per „ETF‑Sampling“:
$$
j \sim \mathbf{q}(t)\quad\Rightarrow\quad m_{ij}\leftarrow 1
$$
Damit bleibt das Modell weltagnostisch und code‑nah.

---

## 11. Hinweise zu Stackbarkeit (Layer als Kapitalmärkte)

„Stack“ = mehrere Ebenen von Märkten/Neuronen, die ineinander investieren.

Kanonische Sicht:
- Layer $\ell$ hat Neuronen $i^{(\ell)}$ und ein ETF $\mathbf{q}^{(\ell)}$.
- Neuronen aus Layer $\ell+1$ können in Layer $\ell$ investieren (oder umgekehrt), solange die Renditen $\mathbf{r}^{(\ell)}(t)$ definiert sind.

**Wichtig für Stackbarkeit:**  
Das Neuron braucht nur:
1) eine Rendite‑Beobachtung $\mathbf{r}(t)$ seiner investierbaren Menge,  
2) einen Referenz‑ETF $\mathbf{q}(t)$,  
3) seine eigenen Kosten/Statistiken.

Damit ist das Modell **kompositionsfähig**: Ein Layer kann die „Assets“ eines anderen Layers als Black‑Box‑Renditesignale konsumieren.

---

## 12. Minimaler „v1“-Algorithmus (ohne Implementation)

Pro Periode $t$, für jedes Neuron $i$:

1) **Beobachte** $\mathbf{r}(t)$ (Renditen der investierbaren Neuronen).  
2) **Berechne** $R_i^{\text{gross}}(t)=\mathbf{w}_i(t)^\top\mathbf{r}(t)$ und $R^{\text{ETF}}(t)=\mathbf{q}(t)^\top\mathbf{r}(t)$.  
3) **Kosten** $\text{Cost}_i(t)=D_i c_{D,i} + E_i c_{E,i} + b_i$.  
4) **Equity‑Update** $E_i\leftarrow E_i + K_i R_i^{\text{gross}} - \text{Cost}_i$.  
5) **Rolling‑Stats** für $z_i(t)=\Delta_i(t)$ oder $\Delta_i^{\text{net}}(t)$: update $\mu_i,\sigma_i^2$.  
6) **Kreditkondition** $\psi_i$ und **Debt‑Update** $D_i$ (Austrocknung).  
7) **Rebalancing**: update $\mathbf{a}_i$ via exponentiated gradient auf Scores, setze $\mathbf{w}_i=(1-\alpha)\mathbf{q}+\alpha\mathbf{a}_i$.  
8) **Exit/Rebirth** falls $E_i<E_{\min}$ oder $K_i<K_{\min}$.

---

## 13. Design‑Entscheidungen (warum das kanonisch ist)

- **Weltagnostisch:** Es gibt keine Zustände einer Welt, keine Positionen, keine Aktionen außer Portfolio‑Gewichten.
- **Lernen nur über Kapital:** Anpassung erfolgt ausschließlich über Rückflüsse $r_j(t)$, Kosten und Rebalancing.
- **ETF als Gleichgewicht:** Jede Optimierung ist relativ und stabilisiert das System (kein driftendes Nullniveau).
- **Stackbar:** Input/Output sind Rendite‑Signale und Kapital; dadurch ist Komposition über Layer möglich.
- **Sparsity‑fähig:** Maske + Kosten + Simplex‑Updates liefern natürliche dünne Graphen.

---

## 14. Offene Freiheitsgrade für v2 (nur als Marker)

- Präzisere Risikomodelle: Kovarianzmatrix‑Skizze auf maskiertem Support.  
- Endogene Renditeerzeugung $r_i(t)$ aus internen „Welt“-Cashflows (Docking‑World).  
- Unterschiedliche Anleger‑Typen (konservativ/unternehmerisch) als Parameterbündel $(\lambda,\gamma,\ell_{\max},b_\alpha)$.  
- Systemweite Liquiditäts-/Pool‑Politik für Rebirth.
