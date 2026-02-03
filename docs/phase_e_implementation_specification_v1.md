# Phase E – Implementation Specification (v1)

**Project:** GAP / Economy of Neurons  
**Branch:** `phases/phase_e_gap_in_the_sandbox`  
**Status:** Working implementation spec (post-convergence)  

---

## 0. Positioning

Phase E is concerned with **selection-driven structuring**, not with optimisation or intelligence.

The system:

* does not know goals
* does not represent rewards
* does not explore intentionally

Structure emerges **only** through survival under externally imposed conditions.

The external selector is called the **Teacher**.

---

## 1. Separation of Concerns (Binding)

### 1.1 World (Adapter Layer)

The **World** is an interchangeable task environment, e.g.:

* Mazeworld (x, y, N/E/S/W, optional z-depth)
* Binomial World
* Driving World

Properties:

* provides local state and transitions
* produces rewards and costs
* may change over time

The World:

* is *not* the Sediment
* is *not* aware of Brokers or Stacks

---

### 1.2 Agents (Internal System)

The system initially consists of **elementary units** ("neurons"):

* partial perception
* partial action capability
* no identity
* no memory

These units participate in a **market of capabilities**.

---

### 1.3 Brokers

**Brokers** are transient mediators that:

* bind elementary units
* coordinate partial capabilities
* reduce market exposure
* consume resources

Brokers:

* are mortal
* may accumulate capital (budget)
* may mask sub-units
* are selected, not trained

---

### 1.4 Inhibitory Brokers

An **Inhibitory Broker** is a Broker that:

* masks a set of sub-units
* withdraws them from the open market
* presents them as a single functional unit

Purpose:

* reduce variance
* stabilise coordination
* lower operating costs

---

## 2. Sediment (Core Concept)

### 2.1 Definition

**Sediment** is the persistent structural residue of *dead Brokers*.

Sediment:

* is not an object
* is not active
* is not aware
* is not directly accessible

Sediment is **compressed history**.

---

### 2.2 Sediment Representation

Sediment is represented as a **Directed Acyclic Graph (DAG)**:

* nodes = failed mediation / masking configurations
* edges = temporal ordering of failures

The DAG:

* is *not* the World
* is *not* a policy
* is *not* queried by agents

It acts as a **structural filter** on future mediation.

---

### 2.3 Sediment Formation

When a Broker dies:

1. its final mediation configuration is recorded as a node
2. ordering edges are added based on failure sequence
3. no feedback is given to surviving agents

Sediment growth is:

* monotonic
* irreversible
* passive

---

## 3. Market Dynamics

### 3.1 Open Market (Phase 1)

* elementary units offer partial capabilities
* no stable coordination
* high churn

---

### 3.2 Broker Formation (Phase 2)

* Brokers bind compatible units
* successful bindings reduce costs
* masked units leave the open market

Market relevance declines when:

* no better offers exist
* coordination advantages dominate

---

### 3.3 Disruption & Obsolescence (Phase 3)

* new capabilities (e.g. z-depth perception) appear
* previously dominant Brokers become suboptimal
* capital drains
* Brokers die

Their remains form **Sediment**.

---

## 4. Teacher (External Selector)

The **Teacher**:

* is not part of the Stack
* is not part of the World
* does not intervene locally

Teacher actions:

* defines phases
* sets selection pressure
* triggers Rebirth

The system cannot perceive the Teacher.

---

## 5. Rebirth Mechanism

### 5.1 Purpose

Rebirth enables **recursive selection** without internal learning.

Without Rebirth:

* selection happens once
* sediment saturates
* evolution stops

---

### 5.2 Semantics

Rebirth:

* respects existing Sediment
* may vary structural details
* instantiates a new Stack
* preserves deep history

Rebirth operates on **structure**, not behaviour.

---

## 6. Curriculum

The Teacher defines a **curriculum**:

* phase-structured
* selective
* irreversible

Only configurations that survive a phase:

* enter the next phase
* encounter new worlds or constraints

The curriculum is applied:

* globally
* repeatedly
* at multiple scales

---

## 7. Target Behaviour

The system is not optimised for autonomy.

Desired outcome:

* robust, reliable, non-exploratory competence
* "sedated" operation
* high performance within trained niches

Metaphor: **Gazelle**, not Human.

---

## 8. Success Criteria

Phase E is successful if:

* structure emerges without explicit learning
* obsolete structures leave durable sediment
* rebirth enables renewed selection
* behaviour stabilises without awareness

---

## 9. Non-Goals (Explicit)

Phase E does **not** aim to create:

* curiosity
* self-models
* meta-cognition
* general intelligence

---

## 10. Closing

Phase E implements a system that remembers by refusing to forget,
selects without instructing,
and becomes competent without knowing why.

What emerges is not planned.
It is merely allowed.
