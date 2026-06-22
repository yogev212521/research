# Phase 1 — Deterministic Domain Comparison

Compares our **statistical** attention network (`Att_PAM`) against the
verified Table 2 results of *Learning Lifted Action Models from Unsupervised
Visual Traces* (arXiv:2604.19043v2) on the article's **deterministic** domains.

This is **Phase 1** of the work plan in [../PRD_WORK_PLAN.md](../PRD_WORK_PLAN.md).

## Design premise

Our network is statistical: it predicts **proposition truth probabilities**
rather than committing to one deterministic action model. It therefore has
**no MILP layer**. The architecturally fair baseline is the paper's
**"w/o MILP"** column; the **"with MILP"** column is shown only as headroom.
The paper's symbolic **Err** metric is **N/A** to a statistical network.

## Files

| File | Purpose |
|---|---|
| `run_phase1.py` | Training + evaluation + comparison harness |
| `article_baseline.json` | Verified Table 2 numbers (w/o MILP + with MILP) |
| `results/phase1_results.json` | Our results + side-by-side deltas |
| `results/run.log` | Latest run log |

## How it works

1. Builds a `Domain_sim` per domain (reusing the repo's `pddlsim_runner`).
2. Generates pddlsim **token traces** and trains a fresh `Att_PAM`.
3. Evaluates on held-out traces:
   - **State Accuracy** = `(1 - mean|pred_truth - actual_truth|) * 100` over the
     truth slot of each present predicate token.
   - **Action Accuracy** = % of transitions whose predicted action one-hot
     (argmax of the action region) matches the actual action.
4. Prints a side-by-side table vs the paper and saves JSON.

## Usage

```bash
# from the repo root
python3 "phase 1/run_phase1.py" --domains hanoi puzzle8

# quick smoke test
python3 "phase 1/run_phase1.py" --domains hanoi \
    --num-traces 6 --trace-size 4 --iterations 2 --eval-traces 3 --eval-length 6

# all deterministic domains with PDDL in the repo
python3 "phase 1/run_phase1.py" --domains hanoi puzzle8 logistics gripper blockworld
```

### Arguments
| Flag | Default | Meaning |
|---|---|---|
| `--domains` | `hanoi puzzle8` | domains to test |
| `--num-traces` | 40 | training traces per domain |
| `--trace-size` | 5 | actions per training trace |
| `--iterations` | 10 | training iterations |
| `--lr` | 1e-4 | learning rate |
| `--eval-traces` | 10 | held-out evaluation traces |
| `--eval-length` | 20 | actions per evaluation trace |

## Baseline (paper Table 2, verified from PDF)

| Domain | State Acc w/o MILP | Action Acc w/o MILP | State Acc +MILP | Action Acc +MILP |
|---|---|---|---|---|
| Blocksworld (MNIST) | 89.22% | 13.67% | 97.81% | 85.33% |
| Gripper | 86.22% | 7.60% | 100% | 100% |
| Logistics | 99.93% | 99.67% | 99.89% | 99.56% |
| Blocksworld (Synth) | 93.90% | 66.67% | 99.29% | 88.67% |
| Hanoi | 97.15% | 57.60% | 98.55% | 81.40% |
| 8-puzzle | 99.90% | 97.40% | 99.77% | 92.60% |

## Notes & caveats
- Absolute numbers are not expected to match the paper exactly: our data
  generation differs and there is no image pipeline in Phase 1 (token traces only).
- Action accuracy here measures next-action identification from the predicted
  token, which is the closest transferable analogue to the paper's metric.
- Next phases: **Phase 2** adds probabilistic effects (statistical-power test);
  **Phase 3** adds video analysis.
