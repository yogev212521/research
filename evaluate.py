"""
Train & Evaluate script for the attention-based PAM model.
Trains models with increasing amounts of training data, then evaluates
prediction performance at each training length — individually or combined.
Generates line-plot graphs (like Network.py) showing how metrics change
as training data grows.
"""

import torch
import numpy as np
import Model.att_only as att_only
from tqdm import tqdm
from IndexManager import IndexManager
from pddlsim_runner import Domain_sim
import asyncio
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import os

# ── Domain setup (mirrors Network.py) ────────────────────────────────────────

indexMn = IndexManager(start_index=20)


def build_domains() -> dict[str, Domain_sim]:
    """Instantiate all four Domain_sim objects (pred_offsets chain automatically)."""
    logistics = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/logistics_domain.pddl",
        PROBLEM_FILE="./pddlDomains/logistics_problem.pddl",
    )
    blocksworld = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/blockworld_domain.pddl",
        PROBLEM_FILE="./pddlDomains/blockworld_problem.pddl",
        pred_offset=logistics.pred_size,
        action_offset=4,
    )
    gripper = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/gripper_domain.pddl",
        PROBLEM_FILE="./pddlDomains/gripper_problem.pddl",
        pred_offset=logistics.pred_size + blocksworld.pred_size,
        action_offset=8,
    )
    rooms = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/rooms_domain.pddl",
        PROBLEM_FILE="./pddlDomains/rooms_problem.pddl",
        pred_offset=logistics.pred_size + blocksworld.pred_size + gripper.pred_size,
        action_offset=11,
    )
    return {
        "logistics": logistics,
        "blocksworld": blocksworld,
        "gripper": gripper,
        "rooms": rooms,
    }


# ── Core: train with increasing data & evaluate at each step ─────────────────

def train_and_evaluate(
    domains: dict[str, Domain_sim],
    trace_size: int = 5,
    trace_length_eval: int = 20,
    training_iterations: int = 10,
    m_range_start: int = 2,
    m_range_end: int = 10,
):
    """
    For m in [m_range_start .. m_range_end):
      1. Generate m traces per selected domain  (trace_size actions each)
      2. Train a fresh model until convergence    (delta-based stopping)
      3. Evaluate predictions on each domain      (trace_length_eval actions)
      4. Record identifier / truth / instance / MSE

    Returns metric_history, summary list, and the last trained model.
    """
    domain_list = list(domains.values())
    token_size = domain_list[0].token_size
    num_domains = len(domain_list)

    metric_history = {
        d.name: {"identifier": [], "truth": [], "instance": [], "MSE": []}
        for d in domain_list
    }
    summary = []

    for m in range(m_range_start, m_range_end):
        print(f"\n{'='*60}")
        print(f"  Training round {m}  —  {m} traces per domain  "
              f"({m * num_domains} total)")
        print(f"{'='*60}")

        # ── 1. generate training traces ───────────────────────────
        traces = []
        for domain in domain_list:
            for _ in tqdm(range(m), desc=f"  {domain.name} traces"):
                trace = None
                while trace is None or np.shape(trace)[0] != trace_size * 2 - 1:
                    trace = asyncio.run(domain.Generate_trace(trace_size))
                traces.append(trace)

        traces = np.array(traces)
        np.random.shuffle(traces)
        batch_size = 1
        traces = traces.reshape(
            traces.shape[0] // batch_size, batch_size, *traces.shape[1:]
        )

        # ── 2. train a fresh model ───────────────────────────────
        model = att_only.Att_PAM(
            output_dim=token_size,
            embed_dim=512,
            input_dim=token_size,
            head_number=8,
        )
        model.train(
            traces,
            lr=0.00001,
            iterations=training_iterations,
            delta=0.0001 / m,
        )

        # ── 3. evaluate on each domain ───────────────────────────
        for domain in domain_list:
            test_trace = asyncio.run(domain.Generate_trace(trace_length_eval))

            N = 0
            identifier, truth, instance, mse = 0, 0, 0, 0

            for step in range(0, min(len(test_trace) - 1, trace_length_eval * 2), 2):
                current_state = test_trace[step]
                expected_next_state = test_trace[step + 1]
                prediction = model.test(trace=np.array(current_state))[0]

                for i, token in enumerate(expected_next_state, 1):
                    for j in range(len(token)):
                        if token[j] == 1:
                            identifier += abs(
                                prediction[i][j] - expected_next_state[i][j]
                            )
                            if domain.prob and domain.prob.get(j):
                                truth += 0
                            else:
                                truth += abs(
                                    prediction[i][j + 1]
                                    - expected_next_state[i][j + 1]
                                )
                            instance += abs(
                                prediction[i][j + 2]
                                - expected_next_state[i][j + 2]
                            )
                            N += 1
                            mse += np.sqrt(
                                instance ** 2 + truth ** 2 + identifier ** 2
                            )

            id_avg = identifier / N if N else 0
            tr_avg = truth / N if N else 0
            in_avg = instance / N if N else 0
            ms_avg = mse / N if N else 0

            metric_history[domain.name]["identifier"].append(id_avg)
            metric_history[domain.name]["truth"].append(tr_avg)
            metric_history[domain.name]["instance"].append(in_avg)
            metric_history[domain.name]["MSE"].append(ms_avg)
            summary.append((domain.name, id_avg, tr_avg, in_avg, ms_avg))

            print(
                f"  {domain.name:15s}  identifier={id_avg:.4f}  "
                f"truth={tr_avg:.4f}  instance={in_avg:.4f}  MSE={ms_avg:.4f}"
            )

    return metric_history, summary, model


# ── Graph generation (line plots like Network.py) ────────────────────────────

def plot_metrics_over_training(
    metric_history: dict[str, dict],
    m_range_start: int,
    m_range_end: int,
    trace_size: int,
    tag: str = "",
):
    """
    For each domain, produce a line-plot figure (like Network.py):
      x-axis  = number of training states
      y-axis  = loss (identifier / truth / instance)
    """
    os.makedirs("graphs", exist_ok=True)

    for domain_name, metrics in metric_history.items():
        # x-axis: total training states at each round
        # each round m → m traces × (trace_size*2-1) tokens per trace
        m_values = list(range(m_range_start, m_range_end))
        x_states = [m * (trace_size * 2 - 1) for m in m_values]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(x_states, metrics["identifier"], marker="o", label="identifier")
        ax1.plot(x_states, metrics["truth"], marker="o", label="truth")
        ax1.plot(x_states, metrics["instance"], marker="o", label="instance")
        ax1.set_ylabel("Loss (MAE)")
        ax1.set_xlabel("Number of training states")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{tag}{domain_name} — Prediction error vs training length")
        ax1.grid(True)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))

        plt.tight_layout()
        fname = f"eval_{tag}{domain_name}_metrics.png"
        plt.savefig(os.path.join("graphs", fname))
        plt.close(fig)
        print(f"  Graph saved → graphs/{fname}")

    # If multiple domains, also create a combined comparison plot
    if len(metric_history) > 1:
        _plot_combined_comparison(metric_history, m_range_start, m_range_end, trace_size, tag)


def _plot_combined_comparison(
    metric_history: dict[str, dict],
    m_range_start: int,
    m_range_end: int,
    trace_size: int,
    tag: str = "",
):
    """Combined figure: one subplot per metric, all domains overlaid."""
    m_values = list(range(m_range_start, m_range_end))
    x_states = [m * (trace_size * 2 - 1) for m in m_values]
    metric_names = ["identifier", "truth", "instance"]

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)), sharex=True)

    for ax, mname in zip(axes, metric_names):
        for dname, metrics in metric_history.items():
            ax.plot(x_states, metrics[mname], marker="o", label=dname)
        ax.set_ylabel("Loss (MAE)")
        ax.set_title(f"{mname.capitalize()}")
        ax.legend(loc="upper right")
        ax.grid(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

    axes[-1].set_xlabel("Number of training states")
    plt.suptitle(f"{tag}Multipool — Prediction error vs training length", fontsize=13)
    plt.tight_layout()
    fname = f"eval_{tag}multipool_combined.png"
    plt.savefig(os.path.join("graphs", fname))
    plt.close(fig)
    print(f"  Graph saved → graphs/{fname}")


# ── Save results to JSON ────────────────────────────────────────────────────

def save_results_json(summary: list, num_domains: int, filepath: str):
    """Write all (iteration, domain, metrics) records to a JSON file."""
    records = []
    for idx, (domain_name, identifier, truth, instance, mse) in enumerate(summary):
        records.append({
            "iteration": int(np.floor(idx / num_domains)),
            "domain": domain_name,
            "identifier": float(identifier),
            "truth": float(truth),
            "instance": float(instance),
            "MSE": float(mse),
        })

    # Pick a unique filename
    base = filepath
    i = 1
    while os.path.exists(filepath):
        name, ext = os.path.splitext(base)
        filepath = f"{name}_{i}{ext}"
        i += 1

    with open(filepath, "w") as f:
        json.dump(records, f, indent=4)
    print(f"  Results saved → {filepath}")
    return filepath


# ── Interactive CLI ──────────────────────────────────────────────────────────

def interactive_menu():
    """Main interactive loop."""
    print("=" * 60)
    print("  TRAIN & EVALUATE — Performance vs Training Length")
    print("=" * 60)

    # Build all domain objects once
    all_domains = build_domains()
    domain_names = list(all_domains.keys())

    # ── Mode selection ────────────────────────────────────────────
    print("\nEvaluation mode:")
    print("  1. Single domain")
    print("  2. Multipool (multiple domains)")
    mode = input("Choose mode [1/2]: ").strip()

    if mode == "1":
        # ── Single domain ─────────────────────────────────────────
        print("\nAvailable domains:")
        for idx, name in enumerate(domain_names, 1):
            print(f"  {idx}. {name}")
        d_choice = input("Choose domain number: ").strip()
        if d_choice.isdigit() and 1 <= int(d_choice) <= len(domain_names):
            selected_name = domain_names[int(d_choice) - 1]
        else:
            print("Invalid choice, defaulting to logistics.")
            selected_name = "logistics"

        selected = {selected_name: all_domains[selected_name]}
        print(f"\n→ Selected single domain: {selected_name}")

    elif mode == "2":
        # ── Multipool ─────────────────────────────────────────────
        print("\nAvailable domains:")
        for idx, name in enumerate(domain_names, 1):
            print(f"  {idx}. {name}")
        print(f"  {len(domain_names) + 1}. ALL domains")

        choices = input("Enter domain numbers separated by commas (e.g. 1,3) or choose ALL: ").strip()
        if choices == str(len(domain_names) + 1) or choices.lower() == "all":
            selected = all_domains
        else:
            indices = [int(c.strip()) for c in choices.split(",") if c.strip().isdigit()]
            selected = {}
            for i in indices:
                if 1 <= i <= len(domain_names):
                    name = domain_names[i - 1]
                    selected[name] = all_domains[name]
            if not selected:
                print("No valid domains selected, defaulting to all.")
                selected = all_domains

        print(f"\n→ Selected multipool: {', '.join(selected.keys())}")
    else:
        print("Invalid mode, defaulting to all domains.")
        selected = all_domains

    # ── Training / evaluation parameters ──────────────────────────
    print("\n--- Training parameters ---")

    ts = input("Trace size (actions per training trace) [default 5]: ").strip()
    trace_size = int(ts) if ts.isdigit() else 5

    ms = input("Min training rounds (m_start) [default 2]: ").strip()
    m_start = int(ms) if ms.isdigit() else 2

    me = input("Max training rounds (m_end)   [default 10]: ").strip()
    m_end = int(me) if me.isdigit() else 10

    it = input("Training iterations per round [default 10]: ").strip()
    train_iter = int(it) if it.isdigit() else 10

    tl = input("Trace length for evaluation   [default 20]: ").strip()
    trace_length_eval = int(tl) if tl.isdigit() else 20

    # ── Run train & evaluate ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING TRAIN & EVALUATE PIPELINE")
    print(f"  Domains : {', '.join(selected.keys())}")
    print(f"  Rounds  : m = {m_start} .. {m_end - 1}")
    print(f"  Trace sz: {trace_size} actions/trace, eval length: {trace_length_eval}")
    print("=" * 60)

    metric_history, summary, model = train_and_evaluate(
        domains=selected,
        trace_size=trace_size,
        trace_length_eval=trace_length_eval,
        training_iterations=train_iter,
        m_range_start=m_start,
        m_range_end=m_end,
    )

    # ── Generate graphs ───────────────────────────────────────────
    tag = "solo_" if len(selected) == 1 else ""
    plot_metrics_over_training(
        metric_history,
        m_range_start=m_start,
        m_range_end=m_end,
        trace_size=trace_size,
        tag=tag,
    )

    # ── Save results ──────────────────────────────────────────────
    json_path = save_results_json(
        summary,
        num_domains=len(selected),
        filepath="eval_results.json",
    )

    # ── Optionally save model weights ─────────────────────────────
    save_w = input("\nSave trained model weights? [y/N]: ").strip().lower()
    if save_w == "y":
        os.makedirs("Parameters", exist_ok=True)
        names = "_".join(selected.keys())
        weight_path = f"./Parameters/eval_{names}_trained.pth"
        torch.save(model.state_dict(), weight_path)
        print(f"  Weights saved → {weight_path}")

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    # Show last-round metrics per domain
    for dname in selected:
        m = metric_history[dname]
        print(
            f"  {dname:15s}  (last round)  "
            f"identifier={m['identifier'][-1]:.4f}  "
            f"truth={m['truth'][-1]:.4f}  "
            f"instance={m['instance'][-1]:.4f}  "
            f"MSE={m['MSE'][-1]:.4f}"
        )
    print(f"\n  Results : {json_path}")
    print(f"  Graphs  : graphs/")
    print("=" * 60)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    interactive_menu()
