#!/usr/bin/env python3
"""
Phase 1 — Deterministic Domain Comparison
=========================================
Trains the statistical attention network (Att_PAM) on the article's
deterministic domains, then compares State/Action accuracy against the
verified Table 2 baselines from arXiv:2604.19043v2.

Design premise: our network is statistical (predicts proposition
probabilities), so it has NO MILP layer. The architecturally fair
baseline is therefore the paper's "w/o MILP" row.

Reuses the repo's existing pipeline:
  - pddlsim_runner.Domain_sim   (token trace generation)
  - Model.att_only.Att_PAM      (the network)
  - IndexManager                (predicate->token index map)

Metrics:
  - State Accuracy  = (1 - mean|pred_truth - actual_truth|) * 100   (truth slot)
  - Action Accuracy = % of transitions whose predicted action (argmax of the
                      action one-hot region) matches the actual action.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

# ── make parent repo importable ──────────────────────────────────────────────
PHASE1_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE1_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import Model.att_only as att_only           # noqa: E402
from IndexManager import IndexManager        # noqa: E402
from pddlsim_runner import Domain_sim        # noqa: E402

BASELINE_PATH = PHASE1_DIR / "article_baseline.json"
RESULTS_DIR = PHASE1_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Domains available with PDDL in the repo (deterministic).
# action_offset values mirror evaluate.py's convention; for single-domain
# training a fresh model is used so offsets only need to be internally valid.
DOMAIN_FILES = {
    "hanoi": {
        "domain": str(REPO_ROOT / "pddlDomains" / "hanoi_domain.pddl"),
        "problem": str(REPO_ROOT / "pddlDomains" / "hanoi_problem.pddl"),
    },
    "puzzle8": {
        "domain": str(REPO_ROOT / "pddlDomains" / "puzzle8_domain.pddl"),
        "problem": str(REPO_ROOT / "pddlDomains" / "puzzle8_problem.pddl"),
    },
    "logistics": {
        "domain": str(REPO_ROOT / "pddlDomains" / "logistics_domain.pddl"),
        "problem": str(REPO_ROOT / "pddlDomains" / "logistics_problem.pddl"),
    },
    "gripper": {
        "domain": str(REPO_ROOT / "pddlDomains" / "gripper_domain.pddl"),
        "problem": str(REPO_ROOT / "pddlDomains" / "gripper_problem.pddl"),
    },
    "blockworld": {
        "domain": str(REPO_ROOT / "pddlDomains" / "blockworld_domain.pddl"),
        "problem": str(REPO_ROOT / "pddlDomains" / "blockworld_problem.pddl"),
    },
}


def build_domain(name: str) -> Domain_sim:
    """Instantiate a single Domain_sim with a fresh IndexManager (offset 0)."""
    files = DOMAIN_FILES[name]
    idx = IndexManager(start_index=20)
    return Domain_sim(
        indexManager=idx,
        DOMAIN_FILE=files["domain"],
        PROBLEM_FILE=files["problem"],
    )


async def generate_traces(domain: Domain_sim, num_traces: int, trace_size: int):
    """Generate `num_traces` token traces, each of length (trace_size*2 - 1)."""
    target_len = trace_size * 2 - 1
    traces = []
    for _ in range(num_traces):
        trace = None
        attempts = 0
        while (trace is None or np.shape(trace)[0] != target_len) and attempts < 50:
            trace = await domain.Generate_trace(trace_size)
            attempts += 1
        if trace is not None and np.shape(trace)[0] == target_len:
            traces.append(trace)
    return traces


def train_model(traces, token_size: int, iterations: int, lr: float):
    """Train a fresh Att_PAM on the generated traces."""
    arr = np.array(traces)
    np.random.shuffle(arr)
    batch_size = 1
    arr = arr.reshape(arr.shape[0] // batch_size, batch_size, *arr.shape[1:])

    model = att_only.Att_PAM(
        output_dim=token_size,
        embed_dim=512,
        input_dim=token_size,
        head_number=8,
    )
    model.train(arr, lr=lr, iterations=iterations)
    return model


def evaluate_model(model, test_trace, action_space: int):
    """
    Returns (state_accuracy_pct, action_accuracy_pct).

    State accuracy : 1 - mean|pred_truth - actual_truth| over truth slots.
    Action accuracy: fraction of transitions whose predicted action one-hot
                     argmax matches the actual action one-hot argmax.
    """
    truth_abs_err = 0.0
    n_props = 0

    action_correct = 0
    action_total = 0

    for step in range(0, len(test_trace) - 1, 2):
        current_state = test_trace[step]
        expected_next_state = test_trace[step + 1]
        prediction = model.test(trace=np.array(current_state))[0]

        # ── action accuracy: token[0] holds the action one-hot region ──
        if len(prediction) > 0 and len(expected_next_state) > 0:
            pred_action = np.argmax(prediction[0][:action_space])
            true_action = np.argmax(expected_next_state[0][:action_space])
            action_correct += int(pred_action == true_action)
            action_total += 1

        # ── state accuracy: truth slot (j+1) for each present predicate ──
        for i in range(1, min(len(expected_next_state), len(prediction))):
            token = expected_next_state[i]
            for j in range(len(token) - 2):
                if token[j] == 1:  # identifier slot present
                    truth_abs_err += abs(
                        prediction[i][j + 1] - expected_next_state[i][j + 1]
                    )
                    n_props += 1

    state_acc = (1.0 - truth_abs_err / n_props) * 100.0 if n_props else 0.0
    action_acc = (action_correct / action_total) * 100.0 if action_total else 0.0
    return state_acc, action_acc


async def run_domain(name: str, args) -> dict:
    print(f"\n{'='*64}\n  DOMAIN: {name}\n{'='*64}")
    domain = build_domain(name)
    token_size = domain.token_size
    action_space = domain.action_space

    print(f"  predicates(token map): {len(domain.token_map)}  "
          f"token_size={token_size}  action_space={action_space}")

    # 1. training traces
    print(f"  Generating {args.num_traces} training traces "
          f"({args.trace_size} actions each)...")
    train_traces = await generate_traces(domain, args.num_traces, args.trace_size)
    print(f"  Got {len(train_traces)} valid training traces.")

    # 2. train
    print(f"  Training Att_PAM (iterations={args.iterations}, lr={args.lr})...")
    model = train_model(train_traces, token_size, args.iterations, args.lr)

    # 3. evaluate over several held-out test traces
    print(f"  Evaluating on {args.eval_traces} held-out traces...")
    state_accs, action_accs = [], []
    for _ in range(args.eval_traces):
        test_trace = await domain.Generate_trace(args.eval_length)
        if not test_trace or len(test_trace) < 2:
            continue
        s_acc, a_acc = evaluate_model(model, test_trace, action_space)
        state_accs.append(s_acc)
        action_accs.append(a_acc)

    state_acc = float(np.mean(state_accs)) if state_accs else 0.0
    action_acc = float(np.mean(action_accs)) if action_accs else 0.0
    state_std = float(np.std(state_accs)) if state_accs else 0.0
    action_std = float(np.std(action_accs)) if action_accs else 0.0

    print(f"  → State Acc = {state_acc:.2f}% (±{state_std:.2f})   "
          f"Action Acc = {action_acc:.2f}% (±{action_std:.2f})")

    return {
        "state_accuracy": state_acc,
        "state_accuracy_std": state_std,
        "action_accuracy": action_acc,
        "action_accuracy_std": action_std,
        "num_train_traces": len(train_traces),
        "eval_traces": len(state_accs),
    }


# Maps a repo domain name -> baseline JSON key (article Table 2).
BASELINE_KEY = {
    "hanoi": "hanoi",
    "puzzle8": "puzzle8",
    "logistics": "logistics",
    "gripper": "gripper",
    "blockworld": "blocksworld_mnist",  # repo grid blocksworld ~ paper MNIST grid
}


def build_comparison(name: str, ours: dict, baseline: dict) -> dict:
    dom = baseline["domains"].get(BASELINE_KEY.get(name, name))
    cmp = {"ours": ours}
    if dom:
        wo = dom["wo_milp"]
        wi = dom["with_milp"]
        cmp["paper_wo_milp"] = wo
        cmp["paper_with_milp"] = wi
        cmp["delta_vs_wo_milp"] = {
            "StateAcc": round(ours["state_accuracy"] - wo["StateAcc"], 2),
            "ActionAcc": round(ours["action_accuracy"] - wo["ActionAcc"], 2),
        }
    return cmp


def print_table(results: dict, baseline: dict):
    print(f"\n\n{'='*88}")
    print("  PHASE 1 — STATE/ACTION ACCURACY vs ARTICLE (Table 2)")
    print(f"{'='*88}")
    header = (f"{'Domain':16} | {'Ours State':>11} {'Ours Act':>9} | "
              f"{'w/o State':>9} {'w/o Act':>8} | {'+MILP State':>11} {'+MILP Act':>9}")
    print(header)
    print("-" * len(header))
    for name, cmp in results.items():
        o = cmp["ours"]
        wo = cmp.get("paper_wo_milp", {})
        wi = cmp.get("paper_with_milp", {})
        print(f"{name:16} | "
              f"{o['state_accuracy']:>10.2f}% {o['action_accuracy']:>8.2f}% | "
              f"{wo.get('StateAcc', float('nan')):>8.2f}% {wo.get('ActionAcc', float('nan')):>7.2f}% | "
              f"{wi.get('StateAcc', float('nan')):>10.2f}% {wi.get('ActionAcc', float('nan')):>8.2f}%")
    print("-" * len(header))
    print("Baseline = paper Table 2. Fair comparison is vs the 'w/o MILP' columns")
    print("(our network is statistical and has no MILP layer). 'Err' is N/A to us.")


async def main():
    parser = argparse.ArgumentParser(description="Phase 1 deterministic comparison")
    parser.add_argument("--domains", nargs="+", default=["hanoi", "puzzle8"],
                        help="domains to test (default: hanoi puzzle8)")
    parser.add_argument("--num-traces", type=int, default=40,
                        help="training traces per domain")
    parser.add_argument("--trace-size", type=int, default=5,
                        help="actions per training trace")
    parser.add_argument("--iterations", type=int, default=10,
                        help="training iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--eval-traces", type=int, default=10,
                        help="held-out traces for evaluation")
    parser.add_argument("--eval-length", type=int, default=20,
                        help="actions per evaluation trace")
    args = parser.parse_args()

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    results = {}
    for name in args.domains:
        if name not in DOMAIN_FILES:
            print(f"[skip] unknown domain '{name}'")
            continue
        try:
            ours = await run_domain(name, args)
            results[name] = build_comparison(name, ours, baseline)
        except Exception as e:
            import traceback
            print(f"[ERROR] domain {name}: {e}")
            traceback.print_exc()

    print_table(results, baseline)

    out = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "baseline_source": baseline["_source"],
        "results": results,
    }
    out_path = RESULTS_DIR / "phase1_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
