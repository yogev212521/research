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
import signal
import atexit
import sys
from multiprocessing import resource_tracker

# ── Proper cleanup handlers ──────────────────────────────────────────────────

def cleanup_resources():
    """Clean up resources on exit."""
    try:
        # Suppress resource tracker warnings during cleanup
        resource_tracker.unregister = lambda *args: None
    except:
        pass
    # Close any matplotlib figures
    plt.close('all')

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    print(f"\n\n{'='*60}")
    print("  Received interrupt signal, cleaning up...")
    print(f"{'='*60}\n")
    cleanup_resources()
    sys.exit(0)

# Register cleanup handlers
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Suppress resource_tracker warnings about leaked semaphores
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

indexMn = IndexManager(start_index=20)


def build_domains() -> dict[str, Domain_sim]:
    """Instantiate all six Domain_sim objects (pred_offsets chain automatically)."""
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
    hanoi = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/hanoi_domain.pddl",
        PROBLEM_FILE="./pddlDomains/hanoi_problem.pddl",
        pred_offset=logistics.pred_size + blocksworld.pred_size + gripper.pred_size + rooms.pred_size,
        action_offset=14,
    )
    puzzle8 = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/puzzle8_domain.pddl",
        PROBLEM_FILE="./pddlDomains/puzzle8_problem.pddl",
        pred_offset=logistics.pred_size + blocksworld.pred_size + gripper.pred_size + rooms.pred_size + hanoi.pred_size,
        action_offset=15,
    )
    return {
        "logistics": logistics,
        "blockworld": blocksworld,
        "gripper": gripper,
        "rooms": rooms,
        "hanoi": hanoi,
        "puzzle8": puzzle8,
    }


# ── Core: train with increasing data & evaluate at each step ─────────────────

async def train_and_evaluate(
    domains: dict[str, Domain_sim],
    trace_size: int = 5,
    trace_length_eval: int = 20,
    training_iterations: int = 10,
    m_range_start: int = 2,
    m_range_end: int = 10,
    repetitions: int = 1,
):
    """
    For m in [m_range_start .. m_range_end):
      For rep in [0 .. repetitions):
        1. Generate m traces per selected domain  (trace_size actions each)
        2. Train a fresh model until convergence    (delta-based stopping)
        3. Evaluate predictions on each domain      (trace_length_eval actions)
        4. Record identifier / truth / instance / MSE

    Each training round m is repeated `repetitions` times.
    All individual data points are stored so graphs display every run.

    Returns metric_history, summary list, and the last trained model.
    """
    domain_list = list(domains.values())
    token_size = domain_list[0].token_size
    num_domains = len(domain_list)

    # Save each domain's probabilistic predicates dict BEFORE Generate_trace
    # overwrites domain.prob with a boolean
    prob_dicts = {}
    for d in domain_list:
        prob_dicts[d.name] = dict(d.prob) if isinstance(d.prob, dict) else {}

    metric_history = {
        d.name: {"identifier": [], "truth": [], "instance": [], "MSE": [], "m_values": []}
        for d in domain_list
    }
    summary = []

    for m in range(m_range_start, m_range_end):
      for rep in range(repetitions):
        rep_label = f" (rep {rep+1}/{repetitions})" if repetitions > 1 else ""
        print(f"\n{'='*60}")
        print(f"  Training round {m}{rep_label}  —  {m} traces per domain  "
              f"({m * num_domains} total)")
        print(f"{'='*60}")

        # ── 1. generate training traces ───────────────────────────
        traces = []
        for domain in domain_list:
            for _ in tqdm(range(m), desc=f"  {domain.name} traces"):
                trace = None
                while trace is None or np.shape(trace)[0] != trace_size * 2 - 1:
                    trace = await domain.Generate_trace(trace_size)
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
            lr=0.000001/m,
            iterations=training_iterations,
            delta=0.00005 / m,
        )

        # ── 3. evaluate on each domain ───────────────────────────
        for domain in domain_list:
            test_trace =  await domain.Generate_trace(trace_length_eval)
            domain_prob = prob_dicts[domain.name]  # use saved prob dict

            N = 0
            identifier, truth, instance, mse = 0, 0, 0, 0
            num_tokens = len(test_trace[0]) if test_trace else 0

            for step in range(0, min(len(test_trace) - 1, trace_length_eval * 2), 2):
                current_state = test_trace[step]
                expected_next_state = test_trace[step + 1]
                prediction = model.test(trace=np.array(current_state))[0]

                # Start at index 1 (skip action token), stop before last to
                # avoid out-of-bounds when accessing prediction[i]
                for i in range(1, min(len(expected_next_state), len(prediction))):
                    token = expected_next_state[i]
                    for j in range(len(token) - 2):  # -2 to safely access j+1, j+2
                        if token[j] == 1:
                            identifier += abs(
                                prediction[i][j] - expected_next_state[i][j]
                            )
                            if domain_prob and domain_prob.get(j):
                                # Compare against known probability instead of
                                # the single-sample outcome
                                truth += abs(
                                    prediction[i][j + 1] - domain_prob[j]
                                )
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
            metric_history[domain.name]["m_values"].append(m)
            summary.append((domain.name, id_avg, tr_avg, in_avg, ms_avg, m, rep))

            print(
                f"  {domain.name:15s}  identifier={id_avg:.4f}  "
                f"truth={tr_avg:.4f}  instance={in_avg:.4f}  MSE={ms_avg:.4f}"
            )

    return metric_history, summary, model


# ── Graph generation (line plots like Network.py) ────────────────────────────

def _compute_avg_per_m(m_vals, values, trace_size):
    """Group data points by m value, compute average and std per m,
    return sorted x, y_avg, y_std.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for m, v in zip(m_vals, values):
        groups[m].append(v)
    sorted_ms = sorted(groups.keys())
    x_avg = [m * (trace_size * 2 - 1) for m in sorted_ms]
    y_avg = [np.mean(groups[m]) for m in sorted_ms]
    y_std = [np.std(groups[m]) for m in sorted_ms]
    return x_avg, y_avg, y_std


def plot_metrics_over_training(
    metric_history: dict[str, dict],
    m_range_start: int,
    m_range_end: int,
    trace_size: int,
    tag: str = "",
):
    """
    For each domain, produce a figure:
      x-axis  = number of training states
      y-axis  = loss (identifier / truth / instance)
    All individual data points are shown as scatter dots.
    If repetitions > 1, an average line is also drawn.
    """
    os.makedirs("graphs", exist_ok=True)

    for domain_name, metrics in metric_history.items():
        # Each domain gets its own subfolder
        domain_dir = os.path.join("graphs", domain_name)
        os.makedirs(domain_dir, exist_ok=True)

        m_vals = metrics["m_values"]
        # x position for each individual data point
        x_all = [m * (trace_size * 2 - 1) for m in m_vals]
        has_reps = len(m_vals) > len(set(m_vals))  # more points than unique m values

        fig, ax1 = plt.subplots(figsize=(10, 6))

        colors = {"identifier": "tab:blue", "truth": "tab:orange", "instance": "tab:green"}
        for mname, color in colors.items():
            # Plot only the average line and ±1σ shaded band (no individual-run points)
            x_avg, y_avg, y_std = _compute_avg_per_m(m_vals, metrics[mname], trace_size)
            y_avg = np.array(y_avg)
            y_std = np.array(y_std)
            ax1.plot(x_avg, y_avg, marker="o", color=color, linewidth=2, label=f"{mname} (avg)")
            ax1.fill_between(x_avg, y_avg - y_std, y_avg + y_std, color=color, alpha=0.15, linewidth=0)

        ax1.set_ylabel("Loss (MAE)")
        ax1.set_xlabel("Number of training states")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{tag}{domain_name} — Prediction error vs training length")
        ax1.grid(True)
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=15))

        plt.tight_layout()
        fname = f"eval_{tag}{domain_name}_metrics.png"
        plt.savefig(os.path.join(domain_dir, fname))
        plt.close(fig)
        print(f"  Graph saved → {domain_dir}/{fname}")

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
    """Combined figure: one subplot per metric, all domains overlaid.
    Shows scatter for individual runs and average lines when repetitions > 1."""
    metric_names = ["identifier", "truth", "instance"]

    fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)), sharex=True)

    for ax, mname in zip(axes, metric_names):
        for dname, metrics in metric_history.items():
            m_vals = metrics["m_values"]
            x_all = [m * (trace_size * 2 - 1) for m in m_vals]
            has_reps = len(m_vals) > len(set(m_vals))

            # Plot only the average line and ±1σ shaded band for each domain
            x_avg, y_avg, y_std = _compute_avg_per_m(m_vals, metrics[mname], trace_size)
            y_avg = np.array(y_avg)
            y_std = np.array(y_std)
            ax.plot(x_avg, y_avg, marker="o", linewidth=2, label=f"{dname} (avg)")
            ax.fill_between(x_avg, y_avg - y_std, y_avg + y_std, alpha=0.12)

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
    for idx, (domain_name, identifier, truth, instance, mse, m_val, rep) in enumerate(summary):
        records.append({
            "round": int(m_val),
            "repetition": int(rep),
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

async def interactive_menu():
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

    rp = input("Repetitions per training round [default 1]: ").strip()
    repetitions = int(rp) if rp.isdigit() and int(rp) >= 1 else 1

    # ── Run train & evaluate ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STARTING TRAIN & EVALUATE PIPELINE")
    print(f"  Domains : {', '.join(selected.keys())}")
    print(f"  Rounds  : m = {m_start} .. {m_end - 1}")
    print(f"  Reps    : {repetitions} per round")
    print(f"  Trace sz: {trace_size} actions/trace, eval length: {trace_length_eval}")
    print("=" * 60)

    metric_history, summary, model = await train_and_evaluate(
        domains=selected,
        trace_size=trace_size,
        trace_length_eval=trace_length_eval,
        training_iterations=train_iter,
        m_range_start=m_start,
        m_range_end=m_end,
        repetitions=repetitions,
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
    # Show last-round metrics per domain (use metric_history keys which match domain.name)
    for dname, met in metric_history.items():
        print(
            f"  {dname:15s}  (last round)  "
            f"identifier={met['identifier'][-1]:.4f}  "
            f"truth={met['truth'][-1]:.4f}  "
            f"instance={met['instance'][-1]:.4f}  "
            f"MSE={met['MSE'][-1]:.4f}"
        )
    print(f"\n  Results : {json_path}")
    print(f"  Graphs  : graphs/")
    print("=" * 60)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        asyncio.run(interactive_menu())
    except KeyboardInterrupt:
        print(f"\n\n{'='*60}")
        print("  Received keyboard interrupt, cleaning up...")
        print(f"{'='*60}\n")
        cleanup_resources()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{'='*60}")
        print(f"  Error occurred: {e}")
        print("  Cleaning up resources...")
        print(f"{'='*60}\n")
        cleanup_resources()
        raise
    finally:
        cleanup_resources()
