# PRD: MILP Consistency Layer for Action-Model Learning

**Source paper:** *Learning Lifted Action Models from Unsupervised Visual Traces* — Kai Xi, Stephen Gould, Sylvie Thiébaux (ICAPS 2026, arXiv:2604.19043v2)
**Document purpose:** Explain the MILP (Mixed-Integer Linear Program) layer so it can be understood, evaluated, and (optionally) re-implemented on top of the existing `Att_PAM` neural pipeline in this repo.

---

## 1. Problem & Motivation

### 1.1 The learning setting
The framework jointly learns three things from sequences of **state images only** (no action labels):
1. **State prediction** — what propositions are true in each frame.
2. **Action prediction** — which action was executed between frames.
3. **A lifted action model** (ROSAME) — preconditions, add-effects, delete-effects per action schema.

### 1.2 Why a pure neural approach fails
Because all three are learned together and unsupervised, the neural model is prone to two failure modes:
- **Prediction collapse** — degenerate solutions where predictions trivially agree but are wrong.
- **Self-reinforcing errors** — early mistakes get amplified as the three predictors reinforce each other, trapping training in local optima.

> The neural losses only enforce *soft* agreement between consecutive states; they do **not** guarantee the predictions are **logically consistent** with a valid planning model.

### 1.3 The MILP's role
The MILP is an **external source of logical consistency**. It takes the neural network's predicted states, actions, and action model over a **small subset of traces**, and solves for the **logically consistent** assignment that is **as close as possible** to the neural predictions. Pseudo-labels extracted from that solution then supervise further neural training — a feedback loop that lets the model **escape local optima and correct its own errors**.

---

## 2. Where the MILP Fits in the Pipeline

```
        ┌──────────────────────────── Deep Learning Framework (PyTorch) ────────────────────────────┐
        │                                                                                            │
 images │   State        Action         ROSAME (lifted                                               │
 ─────▶ │   Predictor    Predictor       action model)                                               │
        │      │            │                 │                                                      │
        │      ▼            ▼                 ▼                                                       │
        │   predicted    predicted        predicted precond/                                         │
        │    states       actions          add/del effects                                           │
        └───────┬────────────┬─────────────────┬─────────────────────────────────────────────────────┘
                │            │                 │
                ▼            ▼                 ▼
        ┌────────────────────────────────────────────────────────────────┐
        │  MILP (Gurobi 12.0.1) — runs on a SUBSET of traces             │
        │  • variables: consistent states, actions, action model        │
        │  • constraints: planning logic (precond holds, effects apply)  │
        │  • objective: stay close to neural predictions                 │
        └───────────────────────────────┬────────────────────────────────┘
                                         │  solution
                                         ▼
                      ┌─────────────────────────────────────┐
                      │  Pseudo-labels (aged, decayed ψ^Δe)  │
                      └──────────────────┬──────────────────┘
                                         │  extra supervised loss
                                         ▼
                          back into the neural training loop
```

---

## 3. How the MILP Works

### 3.1 Inputs
For a sampled subset of traces, the MILP receives the current neural predictions:
- **Predicted states** `pŝₜ` for each time step `t`.
- **Predicted actions** `âₜ` between consecutive states.
- **Predicted action model** `M̂` (lifted precond/add/del per schema, from ROSAME).

### 3.2 Decision variables
The MILP solves for **logically consistent** versions of the same quantities:
- consistent state propositions per step,
- consistent action selection per transition,
- consistent lifted action model.

### 3.3 Constraints (the "logic")
The constraints encode classical-planning validity, e.g.:
- An action may only be selected at step `t` if its **preconditions hold** in state `t`.
- The next state must equal the current state **updated by the action's add/delete effects** (frame consistency).
- Lifted model consistency: all groundings of a schema share the same lifted precond/effect pattern.

These hard constraints are exactly what the soft neural losses cannot guarantee.

### 3.4 Objective
Minimize the **distance to the neural predictions** while satisfying the constraints — i.e., find the *closest logically valid* explanation of what the network believes.

The objective is **modular**: it can include **all, some, or weighted combinations** of three terms:
- **State** term,
- **Action** term,
- **Model** term.

The ablation (paper Table 4) shows the **State + Action + Model** combination is what drives errors to 0 on the hard domains. Using only the State term is much weaker (e.g., Blocksworld synthesized: 8 errors with State-only vs 0 with all three).

---

## 4. Pseudo-Labels & Aging (Decay)

The MILP solution is converted to **pseudo-labels** that act as supervised targets in subsequent epochs.

### 4.1 Why aging is needed
- Early in training, predictions are noisy; a MILP solved on a small subset may **not generalize** across the full dataset.
- MILP solutions from **different epochs may be mutually inconsistent**.

### 4.2 Exponential decay schedule
If a pseudo-label is created at epoch `e₀`, then at a later epoch `e` its loss contribution is weighted by:

$$
w = \psi^{\,(e - e_0)}, \qquad \psi < 1
$$

- Older pseudo-labels **gradually lose influence**.
- Newer ones (from more accurate predictions) are **emphasized**.
- When a trace is **re-sampled**, its pseudo-labels are **recomputed** and replace the old ones.

The paper uses **ψ ≈ 0.99**.

---

## 5. Verified Results (Paper Table 2)

Metrics: **Err** = action-model errors, **Agree** = agreement score, **State Acc**, **Action Acc**.

| Domain | Variant | Err | Agree | State Acc | Action Acc |
|---|---|---|---|---|---|
| Blocksworld (MNIST grid) | w/o MILP | 10 | 0.784 | 89.22% | 13.67% |
| Blocksworld (MNIST grid) | **with MILP** | **0** | **0.977** | **97.81%** | **85.33%** |
| Gripper | w/o MILP | 6 | 0.724 | 86.22% | 7.60% |
| Gripper | **with MILP** | **0** | **0.978** | **100%** | **100%** |
| Logistics | w/o MILP | 0 | 0.979 | 99.93% | 99.67% |
| Logistics | with MILP | 0 | 0.983 | 99.89% | 99.56% |
| Blocksworld (Synthesized) | w/o MILP | 4 | 0.899 | 93.90% | 66.67% |
| Blocksworld (Synthesized) | **with MILP** | **0** | **0.976** | **99.29%** | **88.67%** |
| Hanoi | w/o MILP | 0 | 0.926 | 97.15% | 57.60% |
| Hanoi | **with MILP** | **0** | **0.940** | **98.55%** | **81.40%** |
| 8-puzzle | w/o MILP | 0 | 0.985 | 99.90% | 97.40% |
| 8-puzzle | with MILP | 0 | 0.985 | 99.77% | 92.60% |

**Reading the table:**
- MILP is **essential** for the hard visual domains (Blocksworld MNIST: action accuracy jumps 13.67% → 85.33%; Gripper: 7.60% → 100%).
- MILP is **near-neutral** for already-easy domains (Logistics, 8-puzzle) — the neural model already converges, so MILP adds little.
- The **headline metric** is **Err → 0**: with MILP, the ground-truth action model is recovered **without error** in every domain.

---

## 6. Implications for This Repo's `Att_PAM`

| Aspect | Article framework | This repo (`Att_PAM`) |
|---|---|---|
| State prediction | Neural | ✅ token-based next-state prediction |
| Action prediction | Neural | ✅ action token in trace encoding |
| Lifted action model | ROSAME | ❌ not explicitly learned |
| Logical consistency | **MILP layer** | ❌ none |
| Pseudo-label aging | ψ decay | ❌ none |

**Key takeaway for comparison:** Our `Att_PAM` corresponds most closely to the **"w/o MILP"** rows — a purely neural predictor with no consistency-correction stage. A fair head-to-head should therefore compare:
- our results **vs the "w/o MILP" baseline** (apples-to-apples), and optionally
- the **"with MILP" gap** as the headroom a future MILP layer could close.

---

## 7. Optional Future Work: Adding a MILP Layer

If we later replicate the MILP layer on top of `Att_PAM`:
1. **Decode tokens → states/actions/model** for a sampled subset of traces.
2. **Build the MILP** (Gurobi): variables for consistent states/actions/model; planning-logic constraints; closeness objective with selectable State/Action/Model terms.
3. **Extract pseudo-labels** from the solution.
4. **Add a decayed pseudo-label loss** (`ψ^{e-e₀}`, ψ≈0.99) to the training objective; recompute on re-sampling.
5. **Re-evaluate** against the Table 2 "with MILP" rows.

**Dependencies:** Gurobi 12.0.1, PyTorch ≥ 2.6 (paper's setup).

---

*Numbers and mechanism in this document were extracted directly from the verified PDF of arXiv:2604.19043v2 (Tables 1–4 and the Methods/Experiments sections).*
