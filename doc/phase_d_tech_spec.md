# Phase D – Tech-Spec v0.3.1

**Status:** v0.3.1 – implementierbar  
**Bezug:** Phase D – Reparatur & Stabilisierung (Konzept v0.3)

**Prinzipien:**  
semantikfrei · skaleninvariant · nicht-optimierend · keine neuen Akteure

---

## 0. Notation & Glossar (verbindlich)

**Zeitdiskretisierung**  
- `t ∈ {0,1,2,...}`: diskreter Step-Index (Simulations-/Update-Schritt)

**Schwellen / Parameter**  
- `τ`: Kollaps-Schwelle für `wealth` (global oder per-Knoten `τ[node_id]`)  
- `τ_node`: Kollaps-Schwelle für Proxy-Health (per Knoten/Channel)  
- `ρ`: cap_rate (max. Änderungsrate pro Schritt)  
- `m`: cap_magnitude (max. Niveau)  
- `ε`: min_funding (Soft Bail-out Minimum)  
- `α`: ema_alpha (Glättungsfaktor, 0<α≤1)  
- `N`: Fenstergröße für `immortal_flag` (EWMA über N Schritte)  
- `W`: Fenstergröße für Proxy-Health (Rolling Sum)  
- `K`: Anzahl Seeds/Replicates für Testauswertung

**Simplex-Renorm (Definition)**  
- Zielmenge: `w_i ≥ 0` und `∑_{i∈A} w_i = 1` für aktive IDs `A`  
- Renorm wird **nur** angewandt, wenn eine Policy Gewichte verändert (Caps, Bail-out).  
  Lag Injection erfordert keine Renorm.

**Lag-FIFO Warmup (v0, verbindlich)**  
- Standard: FIFO mit Pass-through während des Warmups. D. h. bis `lag_steps` Beobachtungen vorliegen, werden Signale unverzögert weitergereicht (keine künstliche Null-/Baseline-Füllung).  
- Alternative (nicht Standard): deterministische Prefill/Baseline ist zulässig, muss aber dokumentiert werden.

**Offer / alternative Kopplung**  
- „Offer“ = beobachtetes alternatives Kopplungsangebot für einen Knoten (Quelle: Modellinterne Kandidaten/Edges/Attach-Events).  
- „Leeres Orderbuch“ = `alt_coupling_count == 0` über ein Zeitfenster.

---

## 1. Ziel & Nicht-Ziel

### Ziel
Diese Tech-Spec operationalisiert Phase D **minimal**.  
Sie definiert **Schnittstellen, Parameterpunkte, Telemetrie und Tests**, um Reparaturmechanismen
*ohne Optimierung* implementierbar und beobachtbar zu machen.

Reparatur bedeutet hier ausschließlich:
> **Zeitgewinn zur Analyse von Sterbeprozessen.**

### Nicht-Ziel
- Leistungssteigerung
- Fairness- oder Effizienzkorrektur
- neue Lernregeln oder Agenten
- Governance, Normen oder Policy-Optimierung
- Marktlevel, Marktindex oder Rollenlogik

---

## 2. Begriffe (präzisiert)

- **RepairPolicy**  
  Deterministische, reversible Funktion, die Flüsse verzögert oder begrenzt, ohne neue Information einzuführen.

- **Brokerpfad**  
  Emergenz von Inhibitions-/Risikotransportpfaden.  
  *Kein Akteur, kein Stack-Typ, keine Rolle.*

- **Meta-Kredit**  
  Zeitliche Verschiebung von Risiko ohne Tilgung.

- **Kollaps**  
  Zustand, in dem ein Knoten seine Fortsetzbarkeit verliert:
  `wealth < τ` (oder Proxy-Health unter Schwelle).

- **Fehlende Nachfrage**  
  Zustand ohne alternative Kopplungsangebote (leeres Orderbuch).  
  *Telemetrie, keine Entscheidungsregel.*

---

## 3. Invarianten (Phase D – muss gelten)

1. **Determinismus:** gleiche Inputs + gleicher Startzustand ⇒ gleiche Outputs  
   - Lag-Puffer (FIFO/EMA) sind Teil des Systemzustands und müssen im State enthalten sein.

2. **Keine Historie außerhalb definierter Puffer:**  
   - Historie nur via `lag_steps` FIFO oder EMA-State `r̃_t` (explizit).

3. **Massenerhaltung / Gewichts-Summe:**  
   - Caps & Isolation dürfen keine Masse „erzeugen“.  
   - Soft Bail-out **bricht** Massenerhaltung explizit (Zeitaufschub durch Zuführung).  
   - Nach Caps/Bail-out: **Simplex-Renorm** (Definition siehe Notation).

4. **Keine neue Information:**  
   - Lag/EMA dürfen nur Verzögerung/Glättung sein (keine Zusatzfeatures, keine Lookahead).

---

## 4. RepairPolicy API

### 4.1 Abstraktion

```text
interface RepairPolicy {
  enabled: bool
  params: Map<String, Any>
  apply(state, t) -> state'
}
```

**Eigenschaften:**
- deterministisch
- lokal wirksam
- zur Laufzeit ON/OFF schaltbar
- vollständig reversibel
- keine Speicherung von Historie außerhalb expliziter Lag-Puffer

**Warnhinweis:**  
Keine Policy darf Entscheidungen treffen oder Strategien wählen.

---

### 4.2 Parameterpunkte (v0.3)

#### A) Caps (Flussbegrenzung)

- `cap_rate (ρ)` – |w_t − w_{t−1}| ≤ ρ  
- `cap_magnitude (m)` – w_t ≤ m  

**Nach Anwendung:**  
immer **Simplex-Renorm** auf aktive IDs.

---

#### B) Lag Injection (Zeitentkopplung)

- `lag_steps` – FIFO-Puffer je ID vor `observe()`  
- alternativ `ema_alpha` –  
  r̃_t = α r_t + (1−α) r̃_{t−1}

**Initialbedingungen:**  
- FIFO-Puffer (v0-Default): Pass-through-Warmup (keine Prefill). Erst ab voller Pufferlänge wirkt die Verzögerung.  
- Optional (abweichend): deterministische Prefill/Baseline.  
- EMA: `r̃_0 = r_0` (oder definierte Baseline).

**Wirkung:**  
Entkoppelt Ursache und Wirkung zeitlich, ohne Information zu verändern.

---

#### C) Soft Bail-out (Minimalzuführung)

- `min_funding (ε)` – w_t = max(w_t, ε)  
- `threshold` – Auslöser (`wealth < threshold`)  
- `cooldown` – Sperrzeit

**Nach Anwendung:**  
immer **Simplex-Renorm** auf aktive IDs.

**Hinweis:**  
Soft Bail-out ist Zeitaufschub, keine Rettung (bricht Massenerhaltung explizit).

---

#### D) Isolation (Kopplungsreduktion)

- `isolation_mask`  
  - Phase D minimal: `Set[node_id]`
  - optional: `Set[(src_id, dst_id)]`
- `ttl_steps` – Dauer (optional; `None` = dauerhaft)
- `scope` – lokal / global

**Scheidung:**  
Isolation **ohne TTL** gilt als *dauerhafte Dekopplung*.  
Kein eigener Mechanismus, nur Konfiguration.

---

## 5. Broker Hooks (funktional, nicht-agentisch)

Broker bleiben **Pfadfunktionen**, keine Akteure.

### 5.1 Binding-Points

- `decide_limits()`  
- `apply_policies()`  
- `observe_lag()`

```text
decide_limits(state, t) -> caps/masks
apply_policies(state, policies, t) -> state'
observe_lag(signal, t) -> delayed_signal
```

**Regeln:**
- keine Optimierung
- keine Strategie
- keine Speicherung über explizite Lag-Puffer hinaus

**Warnhinweis:**  
Keine Broker-States, keine Rollen, keine Broker-Stacks einführen.

---

## 6. Telemetrie

### 6.1 Kollaps- und Zustandsdefinition

- **Kollaps-Kriterium:**  
  `wealth < τ` (CapitalSelector.wealth)

- **Kanäle ohne wealth:**  
  Proxy-Health als Rolling Sum über Fenster `W`:  
  `proxy_health_t = ∑_{k=t-W+1}^{t} (r_k − c_k)`  
  Kollaps bei `proxy_health_t < τ_node`

---

### 6.2 Event-Schema (einheitlich)

Alle Events werden als strukturierte Records geloggt:

- `event_id`: UUID oder monotone ID
- `t`: Step-Index
- `event_type`: String
- `subject_id`: `node_id` oder `edge_id`
- `attrs`: JSON (key/value)
- `corr_id`: optional (Kaskaden-/Run-Korrelation)

**ID-Räume:**  
- `node_id`: eindeutige Knoten-ID (String/Int)  
- `edge_id`: eindeutige Kanten-ID oder `(src_id,dst_id)`

---

### 6.3 Events

- `REPAIR_ON(mechanism, params_id)`
- `REPAIR_OFF(mechanism)`
- `NODE_COLLAPSE(cause_hint)`
- `CASCADE_START(seed_node)`
- `CASCADE_END(depth, breadth)`
- `BAILOUT_TRIGGER(amount)`
- `EDGE_GATED(ttl)`
- `EDGE_UNGATED()`

*(t, subject_id, corr_id werden via Schema geführt.)*

---

### 6.4 Counters (Sterbe-Telemetrie)

- `time_to_collapse[node_id]`  
  erstes `t` mit `wealth < τ` (oder Proxy)

- `cascade_depth`
- `cascade_breadth`
- `cascade_count`

- `repair_dependency[node_id]`  
  Δ(time_to_collapse ON–OFF)

- `immortal_flag[node_id]`  
  `surv_EWMA > τ` über `N` Schritte  
  *(N und τ sind verbindlich zu konfigurieren.)*

---

### 6.5 Nachfrage-/Exit-Telemetrie (nur Beobachtung)

**Ziel:** Exploitations-Signaturen sichtbar machen.

- `alt_coupling_count[node_id]`  
- `time_since_last_offer[node_id]`  
- `offer_rate[node_id]` (EWMA über Offers)

**Events (optional):**
- `OFFER_SEEN(offer_id)`
- `NO_OFFER_WINDOW(window)`

> **Ein leeres Orderbuch ist selbst Information.**

**Zählregeln (v0):**
- Duplikate: 동일es `offer_id` zählt pro Fenster nur einmal
- Fenster: Default `window = W_offers` (konfigurierbar)
- Sampling: Source muss deterministisch sein (z. B. Kandidatenliste pro Step)

**Warnhinweis:**  
Diese Größen dürfen **keine** deterministischen Entscheidungen erzwingen.

---

## 7. Tests (A6–A10) – Verbindlichkeit

**Reproduzierbarkeit:**  
- Fixe Seeds
- Zwei Runs (ON/OFF)
- Identischer Startzustand (State Snapshot oder deterministische Re-Init)

**Auswertung über K Seeds:**  
- Standard: `K = 5` Seeds  
- Aggregation: Median (robuster als Mean)  
- Report: Median + IQR (optional)

### A6 – Zeitgewinn ohne Optimierung
**Assert (minimal):**  
`median(time_to_collapse_ON) > median(time_to_collapse_OFF)`

**Praktische Relevanz (optional):**  
`Δ >= Δ_min` (z. B. mindestens 5% der OFF-Laufzeit)

---

### A7 – Fragilitätszuwachs
**Assert (minimal):**  
`median(cascade_depth_ON) ≥ median(cascade_depth_OFF)` **oder**  
`median(cascade_breadth_ON) ≥ median(cascade_breadth_OFF)`

**Praktische Relevanz (optional):**  
Steigerung um mindestens 1 Step (Depth) oder 10% (Breadth), je nach Skala.

---

### A8 – Lag-Signatur
**Peak-Time Definition:**  
`peak_time` = Zeitpunkt maximaler Kaskadenaktivität (z. B. max. collapses per Step)

**Toleranz (verbindlich):**
- diskret: `tolerance = 1` Step (Default)

**Szenario-Anforderung:**  
Erster relevanter Kollaps liegt nach dem Lag-Warmup (d. h. nach `lag_steps`).

**Assert:**  
Akzeptiert werden (Median über Seeds):  
`|Δ − lag_steps| ≤ tolerance` **oder** `|Δ − 2·lag_steps| ≤ tolerance`

**Begründung:**  
Bei FIFO mit Pass-through-Warmup und diskreter Ereignis-Detektion (erster/peak Kollaps) kann die beobachtete Verschiebung **≈ 2·lag_steps** ausfallen. Mit deterministischer Prefill-Variante liegt die Verschiebung **≈ lag_steps**. Beide Fälle sind zulässig, sofern das Warmup konsistent dokumentiert ist.

---

### A9 – Isolation-Trade-off
**Assert:**  
`median(breadth_ON) < median(breadth_OFF)`  
und ggf. `median(depth_ON) ≥ median(depth_OFF)` oder beschleunigter lokaler Kollaps

---

### A10 – Exploitation-Signatur (optional)
**Given:** `alt_coupling_count == 0` über `W_offers` Schritte  
**When:** lokale Bilanz negativ  
**Then:** Bindung bleibt bestehen; Telemetrie zeigt steigende lokale Kosten ohne Exit.

---

## 8. Demos (Minimal)

Je Mechanismus ein reproduzierbarer ON/OFF-Run:

- Caps
- Lag Injection
- Soft Bail-out
- Isolation

**Artefakte (Naming-Convention v0):**
- `logs/phase_d_<mechanism>_<on|off>_seed<k>.jsonl`
- `telemetry/phase_d_<mechanism>_<on|off>_seed<k>.csv`
- `plots/phase_d_<mechanism>_compare.png` (optional)

---

## 9. Abschluss

Diese Tech-Spec implementiert Phase D **ohne** neue Intelligenz,
ohne Governance und ohne Optimierung.

> Phase D endet dort, wo Reparatur sichtbar Zeit kauft –  
> und neue Brüche erzeugt.
