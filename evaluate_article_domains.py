"""
Evaluation script for domains from the research article.
Trains and evaluates on Hanoi and 8-puzzle domains.
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
        resource_tracker.unregister = lambda *args: None
    except:
        pass
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

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

indexMn = IndexManager(start_index=20)


def build_article_domains() -> dict[str, Domain_sim]:
    """Instantiate Hanoi and 8-puzzle domains as described in the article."""
    hanoi = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/hanoi_domain.pddl",
        PROBLEM_FILE="./pddlDomains/hanoi_problem.pddl",
    )
    
    puzzle8 = Domain_sim(
        indexManager=indexMn,
        DOMAIN_FILE="./pddlDomains/puzzle8_domain.pddl",
        PROBLEM_FILE="./pddlDomains/puzzle8_problem.pddl",
        pred_offset=hanoi.pred_size,
        action_offset=1,
    )
    
    return {
        "hanoi": hanoi,
        "puzzle8": puzzle8,
    }


async def train_and_evaluate_article(
    domains: dict[str, Domain_sim],
    trace_size: int = 5,
    trace_length_eval: int = 20,
    training_iterations: int = 10,
    m_range_start: int = 2,
    m_range_end: int = 10,
    repetitions: int = 1,
):
    """
    Train and evaluate on article domains.
    For m in [m_range_start .. m_range_end):
      For rep in [0 .. repetitions):
        1. Generate m traces per domain
        2. Train a fresh model
        3. Evaluate predictions on each domain
        4. Record metrics
    """
    domain_list = list(domains.values())
    token_size = domain_list[0].token_size
    num_domains = len(domain_list)

    # Save probabilistic predicates before they're overwritten
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

            # Generate training traces
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

            # Train a fresh model
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

            # Evaluate on each domain
            for domain in domain_list:
                test_trace = await domain.Generate_trace(trace_length_eval)
                domain_prob = prob_dicts[domain.name]

                N = 0
                identifier, truth, instance, mse = 0, 0, 0, 0
                num_tokens = len(test_trace[0]) if test_trace else 0

                for step in range(0, min(len(test_trace) - 1, trace_length_eval * 2), 2):
                    current_state = test_trace[step]
                    expected_next_state = test_trace[step + 1]
                    prediction = model.test(trace=np.array(current_state))[0]

                    for i in range(1, min(len(expected_next_state), len(prediction))):
                        token = expected_next_state[i]
                        for j in range(len(token) - 2):
                            if token[j] == 1:
                                identifier += abs(
                                    prediction[i][j] - expected_next_state[i][j]
                                )
                                if domain_prob and domain_prob.get(j):
                                    truth += abs(
                                        prediction[i][j + 1] - domain_prob[j]
                                    )
                                else:
                                    truth += abs(
                                        prediction[i][j + 1] - expected_next_state[i][j + 1]
                                    )
                                instance += 1
                                N += 1

                if N > 0:
                    mse = np.sqrt(identifier / N)
                else:
                    mse = 0

                # Record metrics
                metric_history[domain.name]["identifier"].append(identifier)
                metric_history[domain.name]["truth"].append(truth)
                metric_history[domain.name]["instance"].append(instance)
                metric_history[domain.name]["MSE"].append(mse)
                metric_history[domain.name]["m_values"].append(m)

                summary.append(
                    {
                        "domain": domain.name,
                        "m": m,
                        "rep": rep,
                        "MSE": mse,
                        "identifier": identifier,
                        "truth": truth,
                        "instance": instance,
                    }
                )

                print(f"  {domain.name}: MSE={mse:.6f}, identifier={identifier:.2f}, "
                      f"truth={truth:.2f}, instance={instance}")

    return metric_history, summary, model


def plot_learning_curves(metric_history, save_dir="graphs/article_domains"):
    """Generate learning curves for article domains."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot MSE across domains
    plt.figure(figsize=(12, 6))
    for domain_name, metrics in metric_history.items():
        if metrics["m_values"]:
            plt.plot(
                metrics["m_values"],
                metrics["MSE"],
                marker="o",
                label=domain_name,
                linewidth=2,
            )
    plt.xlabel("Training traces (m)", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("Article Domains: Learning Curves", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Individual domain plots
    for domain_name, metrics in metric_history.items():
        if metrics["m_values"]:
            plt.figure(figsize=(10, 5))
            plt.plot(
                metrics["m_values"],
                metrics["MSE"],
                marker="o",
                color="blue",
                linewidth=2,
                markersize=8,
            )
            plt.xlabel("Training traces (m)", fontsize=12)
            plt.ylabel("MSE", fontsize=12)
            plt.title(f"{domain_name.capitalize()}: Learning Curve", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{domain_name}_learning_curve.png", dpi=150, bbox_inches="tight")
            plt.close()


async def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("  Article Domains Evaluation")
    print("  (Hanoi & 8-Puzzle)")
    print("="*60 + "\n")

    domains = build_article_domains()
    
    metric_history, summary, model = await train_and_evaluate_article(
        domains=domains,
        trace_size=5,
        trace_length_eval=20,
        training_iterations=10,
        m_range_start=2,
        m_range_end=8,
        repetitions=1,
    )

    # Save results
    os.makedirs("results", exist_ok=True)
    
    with open("results/article_domains_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("  Results")
    print("="*60)
    for item in summary:
        print(f"  {item['domain']:12s} m={item['m']} rep={item['rep']}: MSE={item['MSE']:.6f}")

    # Generate plots
    plot_learning_curves(metric_history)

    print("\n✓ Evaluation complete!")
    print(f"✓ Results saved to results/article_domains_results.json")
    print(f"✓ Learning curves saved to graphs/article_domains/")


if __name__ == "__main__":
    asyncio.run(main())
