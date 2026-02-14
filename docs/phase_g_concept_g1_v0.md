# Phase G – Concept G1 (v0)

Status: Draft / Conceptual

Scope: **G1 only** – Profile definition & instantiation contract
Normative sources:

* `docs/math-v1.md` (§18, §19)
* `docs/phase_g_concept_v0.md`

---

## 1. Goal of G1

The goal of **G1** is to enable the coexistence of **two operational profiles (A and B)**
*without introducing new semantics*, *without modifying the canonical core*, and *without creating an open configuration space*.

Profile A and Profile B are **not separate models**.
They are **two closed, internally consistent instantiations** of the **same canonical kernel**.

G1 is a **contract-definition phase**, not an optimization or exploration phase.

---

## 2. Definitions

### 2.1 Canonical Kernel

The *canonical kernel* is the complete set of:

* state representation
* update equations
* invariants
* rebirth, drawdown, and freeze logic

The canonical kernel:

* is **profile-agnostic**
* contains **no conditional logic** referring to profiles
* is the sole authority on model semantics

### 2.2 Profile

A *profile* is a **closed configuration bundle** that fully specifies all degrees of freedom
*allowed by* the canonical kernel.

Profiles:

* do **not** add state variables
* do **not** alter update equations
* do **not** weaken or replace invariants
* do **not** introduce alternative dynamics

Profiles exist **only at instantiation time**.

### 2.3 Model Instance

A *model instance* is the result of applying a profile through a builder.

After instantiation:

* the instance contains **no profile identity**
* all configuration choices are fully resolved
* the kernel cannot distinguish whether Profile A or B was used

---

## 3. Normative Profiles (from `math-v1.md`)

### 3.1 Profile A (Canonical)

Profile A corresponds to the **canonical / production profile** defined in `math-v1.md` §18.

Properties:

* semantically binding reference model
* fixed scoring, statistics, freeze, and rebirth behavior
* no use of sediment data for causal dynamics

Profile A defines the **baseline semantics** of the system.

### 3.2 Profile B (Non-canonical, Observational)

Profile B corresponds to the **non-canonical profile** defined in `math-v1.md` §19.

Properties:

* observational / analytical only
* may differ in scoring signals, statistics accumulation, stack weighting, or freeze behavior
* **must not** introduce causal feedback from sedimented data

Profile B is explicitly **not a production profile**.

---

## 4. Allowed vs. Forbidden Differences

### 4.1 Allowed (Profile-level)

Profiles may differ in:

* parameter values
* scoring signal selection (as defined in `math-v1.md`)
* statistics accumulation rules
* stack weighting strategies
* freeze-time observation behavior
* rebirth activation behavior (activation triggers only, e.g. thresholds; reset semantics are invariant and must not differ between profiles)

These differences must be fully resolved **before kernel execution**.

### 4.2 Forbidden (Kernel-level)

Profiles must **never**:

* add or remove state variables
* modify update equations
* introduce profile-conditional branches in the kernel
* alter rebirth or drawdown semantics
* use sediment data causally

Any such change constitutes **new semantics** and is out of scope for Phase G.

---

## 5. Builder Boundary

### 5.1 KernelBuilder

All profile logic is confined to a **single instantiation boundary**:

```
KernelBuilder.fromProfileA(...)
KernelBuilder.fromProfileB(...)
```

The builder:

* selects exactly one closed profile
* constructs a fully resolved configuration object
* produces a model instance bound only to the canonical kernel

### 5.2 Post-instantiation Invariant

After instantiation:

* no profile identity is retained
* no profile checks are possible or permitted
* kernel execution is identical across instances

---

## 6. Equivalence Criterion

For any parameter set where Profile A and Profile B resolve to identical values:

> The resulting model instances **must produce identical behavior**.

This criterion defines the **canonical correctness test for G1**.

---

## 7. Non-goals (Explicit)

G1 does **not** include:

* performance optimization
* CUDA support
* empirical analysis
* learning or adaptation
* architectural refactoring beyond builder isolation

These concerns are addressed in later G-tasks.

---

## 8. Exit Condition for G1

G1 is considered complete when:

* profiles A and B are fully specified as closed configuration bundles
* a single canonical kernel is used for all instances
* profile selection occurs exclusively via the builder
* no profile-specific logic exists in the kernel

Completion of G1 enables G2–G4 without semantic ambiguity.
