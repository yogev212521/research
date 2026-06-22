# PRD & Work Plan: Statistical Action-Model Network vs. Article Benchmarks

**Owner:** Research (NN action-model)
**Network under test:** `Att_PAM` (attention-based, statistical next-state predictor) + `pddlsim_runner` + `IndexManager`
**Reference paper:** *Learning Lifted Action Models from Unsupervised Visual Traces* — Xi, Gould, Thiébaux (ICAPS 2026, arXiv:2604.19043v2)

---

## 1. Vision

Demonstrate that our **statistical** attention network is a competitive and *more general* action-model learner than the article's approach. Unlike the paper — which relies on a MILP layer to enforce a single **deterministic** logically-consistent model — our network predicts **proposition probabilities**, so it natively handles **non-deterministic effects** without symbolic correction.

The program proceeds in three escalating phases:

1. **Match** the article on its own **deterministic** domains.
2. **Exceed** the article's scope by adding **probabilistic effects** and measuring statistical accuracy.
3. **Extend** to perception by analyzing **video** inputs.

---

## 2. Why No MILP (Design Premise)

| | Article (MILP) | Our network |
|---|---|---|
| Objective | One deterministic action model (Err → 0) | Distribution over proposition truth values |
| Consistency | Hard logical projection (MILP) | Soft, probabilistic |
| Effects | Deterministic add/delete | **Deterministic *and* stochastic** |
| Correction layer | Required | **Not needed** |

MILP exists to fix logical contradictions in a deterministic model. Our network estimates likelihoods, so a hard logical projection would destroy the probabilistic signal we want to keep. We therefore compare against the paper's **"w/o MILP"** column (purely neural, no symbolic correction) as the architecturally fair baseline, and treat **State/Action Accuracy** as the transferable metrics (the paper's symbolic **Err** count is **N/A** to a statistical model).

---

## 3. Phase 1 — Deterministic Domain Comparison

**Goal:** Validate the network on the article's deterministic domains and compare State/Action accuracy against the paper's reported numbers.

### Scope
- Domains: start with **Hanoi** and **8-puzzle** (PDDL already in repo), then optionally extend to Blocksworld, Gripper, Logistics.
- Baseline: paper **Table 2 "w/o MILP"** rows (and "with MILP" shown as headroom reference).

### Tasks
1. Build an isolated comparison harness (separate folder) that trains a fresh `Att_PAM` per domain via the existing `evaluate.py` pipeline.
2. Generate pddlsim token traces; train; evaluate on held-out traces.
3. Convert the network's truth-slot MAE into **State Accuracy = 1 − mean|pred − actual|**; compute **Action Accuracy**.
4. Emit a side-by-side table: *our result* vs *paper w/o-MILP* vs *paper with-MILP*.

### Reference baseline (verified from PDF, Table 2)
| Domain | State Acc (w/o MILP) | Action Acc (w/o MILP) | State Acc (with MILP) | Action Acc (with MILP) |
|---|---|---|---|---|
| Blocksworld (MNIST) | 89.22% | 13.67% | 97.81% | 85.33% |
| Gripper | 86.22% | 7.60% | 100% | 100% |
| Logistics | 99.93% | 99.67% | 99.89% | 99.56% |
| Blocksworld (Synth) | 93.90% | 66.67% | 99.29% | 88.67% |
| Hanoi | 97.15% | 57.60% | 98.55% | 81.40% |
| 8-puzzle | 99.90% | 97.40% | 99.77% | 92.60% |

### Success criteria
- Network runs end-to-end on each deterministic domain.
- State accuracy is competitive with the paper's **w/o-MILP** numbers.
- Reproducible results saved as JSON + comparison table.

### Deliverables
- `comparison/` folder with harness, copied domains/problems, baseline JSON, results, and side-by-side report.

---

## 4. Phase 2 — Probabilistic Effects (Statistical Power Test)

**Goal:** Inject **non-deterministic effects** into the domains and show the network learns the underlying **probability distribution** — the capability MILP-based deterministic methods cannot represent.

### Approach
1. **Author probabilistic domain variants** — extend the PDDL/simulator so chosen actions have stochastic outcomes (e.g., an effect applies with probability `p`). The simulator already carries probability hooks (`action_prob`, the `prob` dicts and the `test` predicate in `pddlsim_runner.py`); formalize and parameterize these.
2. **Define ground-truth distributions** per stochastic effect so we know the true `p` to score against.
3. **Train** `Att_PAM` on traces sampled from the stochastic simulator.
4. **Evaluate statistical fit**, not just hard accuracy:
   - Predicted probability vs true `p`: **truth-slot calibration error** = mean|pred_prob − true_p|.
   - Reliability/calibration curve per stochastic predicate.
   - Compare against deterministic-only baselines to show added value.

### Why this matters
This phase tests the **core thesis**: a statistical network captures uncertainty that a deterministic, MILP-corrected model collapses away. Success here is the project's main differentiator.

### Success criteria
- Network's predicted probabilities track the injected `p` within a target tolerance (e.g., mean abs calibration error below an agreed threshold).
- Demonstrated degradation of a deterministic baseline on the same stochastic domains (contrast).

### Deliverables
- Probabilistic domain variants (separate folder).
- Calibration metrics + reliability plots.
- Short results note: statistical accuracy vs ground-truth probabilities.

### Open questions
- Which actions/effects get randomized, and at what probabilities?
- PDDL representation of stochastic effects (custom tags vs probabilistic-PDDL style) and how the simulator samples them.

---

## 5. Phase 3 — Video Analysis

**Goal:** Move from symbolic/token traces toward **perception**, analyzing **video** sequences as input to the pipeline.

### Approach (high level, to be detailed later)
1. Define the video → state-token interface (frames → proposition estimates feeding the existing token encoding).
2. Choose/Build a perception front-end (per-frame state estimator).
3. Feed extracted state tokens into `Att_PAM`; reuse Phases 1–2 evaluation.
4. Evaluate end-to-end accuracy and probability calibration on video-derived traces.

### Success criteria
- End-to-end run from video frames to action-model predictions.
- Accuracy/calibration reported on at least one domain.

### Deliverables
- Video ingestion module + interface spec.
- End-to-end demo on one domain.

### Open questions
- Video source (rendered domain rollouts vs recorded), resolution, frame rate.
- Perception model choice and training data.

---

## 6. Metrics Summary

| Metric | Phase | Meaning |
|---|---|---|
| State Accuracy = 1 − mean\|pred − actual\| | 1, 2, 3 | Per-proposition truth correctness |
| Action Accuracy | 1, 2, 3 | Correct action identification |
| Calibration error = mean\|pred_prob − true_p\| | 2, 3 | Statistical fit to stochastic ground truth |
| Reliability curve | 2, 3 | Predicted vs observed probability |
| (Reference only) Err | 1 | Paper's symbolic model errors — **N/A** to our statistical net |

---

## 7. Phasing & Dependencies

```
Phase 1 (deterministic compare) ──▶ Phase 2 (probabilistic effects) ──▶ Phase 3 (video)
        │                                   │                               │
   needs: PDDL + harness            needs: stochastic sim +           needs: perception
   + verified baselines             ground-truth p                    front-end
```

- **Phase 1** is the foundation: harness + baselines must be solid before adding stochasticity.
- **Phase 2** depends on the Phase-1 evaluation code (reused with calibration metrics added).
- **Phase 3** depends on a stable token interface from Phases 1–2.

---

## 8. Non-Goals
- Implementing a MILP / symbolic consistency layer (explicitly out of scope by design).
- Matching the paper's absolute numbers exactly (different data generation, no image pipeline in Phase 1).
- Learning a single deterministic action model (we model distributions).

---

*Baselines in this plan were extracted from the verified PDF of arXiv:2604.19043v2 (Table 2).*
