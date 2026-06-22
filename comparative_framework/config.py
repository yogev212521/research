"""
Configuration for Comparative Framework
ROSAME Visual Network → Token Prediction → Comparative Analysis
"""

import os
from pathlib import Path

# Root paths
PROJECT_ROOT = Path(__file__).parent.parent
ROSAME_ROOT = Path("/Users/yogevk/Downloads/ROSAME")
COMPARATIVE_ROOT = PROJECT_ROOT / "comparative_framework"

# Data paths
DATA_DIR = COMPARATIVE_ROOT / "data"
RESULTS_DIR = COMPARATIVE_ROOT / "results"
ROSAME_DATA = ROSAME_ROOT / "data"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Domain configuration
DOMAINS = {
    "hanoi": {
        "domain_file": str(PROJECT_ROOT / "pddlDomains" / "hanoi_domain.pddl"),
        "problem_file": str(PROJECT_ROOT / "pddlDomains" / "hanoi_problem.pddl"),
        "data_generator": "generate_hanoi_pddlgym.py",
        "article_baseline": {
            "state_accuracy": 98.55,
            "action_accuracy": 81.40,
            "model_error": 0.00,
            "num_objects": 7,  # 4 discs + 3 pegs
            "num_propositions": 55,
            "num_actions": 120,
        },
    },
    "puzzle8": {
        "domain_file": str(PROJECT_ROOT / "pddlDomains" / "puzzle8_domain.pddl"),
        "problem_file": str(PROJECT_ROOT / "pddlDomains" / "puzzle8_problem.pddl"),
        "data_generator": "generate_slide_pddlgym.py",
        "article_baseline": {
            "state_accuracy": 99.77,
            "action_accuracy": 92.60,
            "model_error": 0.00,
            "num_objects": 9,  # 8 tiles + 1 blank
            "num_propositions": 153,
            "num_actions": 576,
        },
    },
}

# Training configuration
TRAINING_CONFIG = {
    "device": "cpu",  # Use "cuda" if available
    "batch_size": 32,
    "trace_length": 5,
    "num_traces": 100,  # Start small for benchmarking
    "train_test_split": 0.9,
    "seed": 42,
}

# Model configuration
MODEL_CONFIG = {
    "embed_dim": 64,
    "head_number": 8,
    "token_size": 70,
    "num_tokens": 60,
    "action_space": 20,
    "learning_rate": 1e-4,
    "epochs": 100,
}

# ROSAME visual network configuration
ROSAME_CONFIG = {
    "device": "cpu",  # Use "cuda" if available
    "block_dim": (28, 28),  # MNIST grid size
    "block_size": 1,
    "hidden_dim": 256,
    "prop_dim": None,  # Will be set per domain
}

# Comparative test configuration
COMPARISON_METRICS = [
    "state_accuracy",
    "action_accuracy", 
    "model_error",
    "precision",
    "recall",
]

# Logging
VERBOSE = True
LOG_INTERVAL = 10

# File paths within results
RESULTS_FILES = {
    "metrics": "comparative_results.json",
    "prediction_trace": "prediction_trace.json",
    "comparison_plot": "comparison_vs_article.png",
}
