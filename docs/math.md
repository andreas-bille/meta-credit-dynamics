# Meta‑Credit Dynamics – Mathematical Foundations & Executive Overview

This document has **two strictly separated parts**:

* **Part I – Mathematical Foundations** (for reviewers, researchers, engineers)
* **Part II – Executive Overview** (for decision‑makers, product owners, sponsors)

They describe the *same system* at different abstraction levels.

---

# Part I – Mathematical Foundations

## 1. System Primitives

### 1.1 Units (Explorers)

Let $i \in \mathcal{U}$ index atomic units ("neurons", explorers).
Each unit allocates capital weights over available channels.

At time $t$:

$$
w_i(t) \in \Delta^n \quad (\text{simplex})
$$

where:

$$
\Delta^n = \{ w \in \mathbb{R}^n_{\ge 0} \mid \sum_k w_k = 1 \}
$$

Long‑only constraint is enforced by construction.

---

### 1.2 Returns and Costs

Each channel $k$ yields a stochastic return $r_k(t)$.

Unit return:

$$
R_i(t) = \sum_k w_{ik}(t)\, r_k(t)
$$

Costs are explicit and subtractive:

$$
C_i(t) = C_{info} + C_{activity} + C_{aggregation}
$$

Net flow:

$$
\Pi_i(t) = R_i(t) - C_i(t)
$$

---

## 2. Weight Update (Rebalancing)

Weights evolve via **Exponentiated Gradient**:

$$
w_{ik}(t+1) = \frac{w_{ik}(t)\, e^{\eta s_k(t)}}{\sum_j w_{ij}(t)\, e^{\eta s_j(t)}}
$$

where $s_k(t)$ is a channel score derived from recent net flows.

This guarantees:

* positivity
* normalization
* implicit sparsity under poor performance

---

## 3. Statistics and Stability

Each unit maintains exponentially weighted statistics:

* Mean:
  $$
  \mu(t) = (1-\beta)\mu(t-1) + \beta \Pi(t)
  $$

* Variance:
  $$
  \sigma^2(t) = (1-\beta)\sigma^2(t-1) + \beta (\Pi(t)-\mu(t))^2
  $$

* Drawdown:
  $$
  DD(t) = \max_{\tau \le t} \sum_{u=\tau}^t -\Pi(u)
  $$

Stability is a threshold predicate:

$$
{}\text{stable}_i(t) := \bigwedge_j S_j(t) \le \tau_j
$$

---

## 4. Stacks (Mediation Structures)

A stack $S \subset \mathcal{U}$ is an aggregated channel:

$$
R_S(t) = \sum_{i \in S} \alpha_i R_i(t)
$$

with aggregation cost:

$$
C_S(t) = C_{agg} + \sum_{i \in S} C_i(t)
$$

Stacks are **structural identities** exported as new channels.

---

## 5. Dissolution and Death

A stack dissolves at time $t_d$ if:

$$
\neg \text{stable}_S(t_d)
$$

Dissolution is **irreversible**.

---

## 6. Sedimented Mediation Paths (Phase E)

### 6.1 Sediment Node

A failed mediation configuration induces a sediment node:

$$
\nu = (M, P, w, \phi, t_d, r)
$$

where:

* $M$ = set of members
* $P$ = masking signature
* $\phi$ = phase id
* $t_d$ = dissolution time
* $r$ = run id

---

### 6.2 Sediment DAG

Sediment nodes form a DAG:

$$
\mathcal{G} = (\mathcal{V}, \mathcal{E})
$$

with edges defined by temporal order:

$$
(\nu_i \to \nu_j) \iff t_i < t_j \land r_i = r_j
$$

The DAG is **append‑only**.

---

### 6.3 Exclusion Rule

A candidate stack ( M' ) is forbidden iff:

$$
\exists \nu \in \mathcal{V}: M' = M_{\nu} \land \phi' = \phi_{\nu}
$$

No soft penalties, no gradients.

---

## 7. Rebirth

Rebirth resets behavioral state:

$$
{}\text{state}_{t+1} \leftarrow \text{init}
$$

while preserving sediment:

$$
\mathcal{G}_{t+1} = \mathcal{G}_t
$$

This separates **learning** from **history**.

---

## 8. Freeze (Inference Mode)

Freeze disables:

* new unit creation
* sediment insertion
* rebirth

Execution continues on a fixed topology:

$$
\mathcal{G},\; S,\; w \quad \text{constant}
$$

---

## 9. Summary (Formal)

The system learns by **irreversible exclusion**.

Optimization occurs only within the space that has not been structurally ruled out.

No semantics, no global objective, no planning is required.

---

# Part II – Executive Overview

## What Is This?

This project builds **decision systems that learn what must not be done**.

Instead of optimizing endlessly, the system:

* tries many structures,
* permanently discards those that fail,
* and operates reliably within what remains.

---

## Why This Matters

Most AI systems:

* never stop learning,
* can repeat catastrophic mistakes,
* and cannot be safely frozen for deployment.

This system is different:

* failures leave permanent traces,
* learning converges naturally,
* and training can be stopped on purpose.

---

## What Problem Does It Solve?

It is suited for:

* safety‑critical control
* enterprise decision pipelines
* automated coordination systems
* complex rule‑constrained environments

Especially where:

> *Doing the wrong thing even once is unacceptable.*

---

## How Is It Trained?

Through **curricula**, not reward hacking.

The system is exposed to increasingly constrained environments.
Failed internal structures are permanently excluded.

Training ends when failures stop occurring.

---

## What Happens After Training?

The system is **frozen**:

* no further learning
* no exploration
* deterministic, auditable behavior

This makes it deployable.

---

## What It Is Not

It is **not**:

* a general AI
* a planning system
* a language model
* a reinforcement learner

It does not reason about meaning.
It does not optimize goals.

---

## Why This Is New

The key innovation is **structural memory through exclusion**.

History shapes behavior by removing possibilities,
not by encoding knowledge.

This is closer to engineering safety systems
than to artificial cognition.

---

## Bottom Line

This approach trades optimality for:

* reliability
* stability
* and controllability

That makes it suitable for production systems
where failure is costly.
