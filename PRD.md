# Product Requirements Document: Neurosymbolic Action Model Learning

## 1. Executive Summary

This research codebase implements a **neurosymbolic learning system** that learns **action models (PAMs - Precomputed Action Models)** for PDDL planning domains using **attention-based neural networks**. The system combines symbolic PDDL domain representations with deep learning to predict state transitions from action sequences, addressing the gap between symbolic planning and neural learning.

**Key Innovation**: Uses multi-head self-attention transformers to learn how actions transform world states in planning domains, measured across multiple standard benchmark domains (Logistics, Blockworld, Gripper, Rooms).

---

## 2. Research Context

### Problem Statement
- Standard PDDL planners require complete action axioms to generate valid plans
- Learning action effects from experience is challenging in symbolic domains
- A neurosymbolic approach can bridge symbolic knowledge representation with neural pattern learning

### Approach
- **Symbolic Layer**: PDDL domains formally define planning problems (predicates, actions, constraints)
- **Simulation Layer**: `pddlsim` library executes simulations and generates action traces
- **Learning Layer**: Attention-based neural network (Att_PAM) learns state transition patterns
- **Evaluation Layer**: Measures prediction accuracy across domains with variable training data

---

## 3. System Architecture

### 3.1 Core Components

```
┌─ Input Layer ─────────────────────────────────────────┐
│  PDDL Domain + Problem Files (.pddl)                  │
│  ├─ Actions: predicates, preconditions, effects      │
│  ├─ Domain-specific problem instances                 │
│  └─ Multiple domains: logistics, blockworld, gripper │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│ Simulation & Trace Generation (pddlsim_runner.py)     │
│ ├─ LocalSimulator: Executes domain simulations        │
│ ├─ Domain_sim: Wraps PDDL domains for learning        │
│ ├─ Generate_trace(): Creates action sequences         │
│ └─ get_tokens(): Converts states → token vectors      │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│ Token Encoding (IndexManager.py)                       │
│ ├─ Maps predicates to tensor indices                   │
│ ├─ Chains offsets across domains                       │
│ ├─ Creates fixed-size token sequences (60 tokens)      │
│ └─ Token size: 70-dimensional vectors                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│ Neural Learning (Model/att_only.py)                    │
│ ├─ Att_PAM: Multi-head Attention Transformer           │
│ ├─ 4 attention layers with layer normalization         │
│ ├─ Feed-forward networks after each attention layer    │
│ ├─ Input embedding: dim → embed_dim/2 → embed_dim     │
│ └─ Output embedding: embed_dim → embed_dim/2 → dim    │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────┴──────────────────────────────────────┐
│ Evaluation Pipeline (evaluate.py)                      │
│ ├─ Trains with m ∈ [m_start, m_end) traces per domain │
│ ├─ Measures MSE on eval traces                         │
│ ├─ Generates scaling graphs showing learning curves    │
│ ├─ Multi-domain evaluation                             │
│ └─ Results saved to eval_results.json                  │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Key Classes & Modules

| Component | File | Purpose |
|-----------|------|---------|
| **Domain_sim** | `pddlsim_runner.py` | Wraps PDDL domains, manages trace generation, handles token encoding |
| **IndexManager** | `IndexManager.py` | Maps PDDL predicates to neural network token indices across domains |
| **Att_PAM** | `Model/att_only.py` | Multi-head attention transformer for learning state transitions |
| **blockWorld, LogisticsDomain, HanoiTowers** | `Domains/domain_generator.py` | Python implementations of planning domains with action simulators |
| **PDDL Domains** | `pddlDomains/*.pddl` | Formal PDDL specifications of planning problems |
| **Training Pipeline** | `evaluate.py` | Main training loop: generates traces → trains model → evaluates |
| **Evaluation** | `evaluate.py` | Computes metrics (MSE, accuracy), generates learning curve graphs |

---

## 4. Data Flow & Pipeline

### 4.1 Trace Generation Pipeline

**Input**: PDDL Domain + Problem definition  
**Output**: Sequence of state-action pairs (traces)

```
1. Load PDDL domain & problem files
   ↓
2. Create LocalSimulator with domain configuration
   ↓
3. Execute random policy for N steps (max_steps=trace_size)
   ↓
4. For each step:
   - Get current perceived state
   - Get applicable (grounded) actions
   - Pick random action
   - Record action as token vector
   - Append affected predicates as tokens
   ↓
5. Generate fixed-size token sequence (pad to 60 tokens of 70 dims each)
   ↓
Output: Trace of shape (trace_length, num_tokens, token_size)
```

### 4.2 Token Encoding Scheme

**Three token types per trace**:

1. **Action Token** (first token)
   - One-hot encoding of action name
   - Indices: [0, action_offset) per domain
   - Example: "move" in logistics at index 2

2. **Object Predicate Tokens**
   - Encode predicates grounded in action arguments
   - Each predicate: 3-dim encoding (flag + 2 state dims)
   - Base index from IndexManager + domain offset
   - Example: `at(package1, loc2)` at index 20+domain_offset

3. **Padding Tokens**
   - Zero-filled vectors to pad to fixed length (60 tokens)
   - Allows batching variable-length sequences

**Multi-domain Indexing**:
- Each domain chains offsets to avoid conflicts
- Logistics: pred_offset=0, action_offset=0
- Blockworld: pred_offset=logistics_pred_size, action_offset=4
- Gripper: pred_offset=prev_total, action_offset=8
- Rooms: pred_offset=prev_total, action_offset=11

### 4.3 Training Loop

**Input**: m traces per domain (4 domains × m = 4m total traces)  
**Process**:

```
For m in [m_start, m_end):
  For each repetition:
    1. Generate m traces from each of 4 domains (4m total)
    2. Concatenate traces into training batch
    3. Initialize fresh Att_PAM model
    4. Train with adaptive learning rate until convergence
       - Uses MSE loss
       - Delta-based stopping (loss change < threshold)
    5. For each domain:
       a. Generate eval_traces (trace_length_eval actions)
       b. Run inference: model.test(trace)
       c. Compute MSE vs. ground truth
       d. Record metric: [identifier, truth, instance, MSE]
    6. Save results to eval_results.json and domain_results.json
    7. Generate learning curve graphs
```

---

## 5. Benchmark Domains

### 5.1 Domain Specifications

| Domain | Files | Objects | Actions | Predicates | Purpose |
|--------|-------|---------|---------|-----------|---------|
| **Logistics** | `logistics_domain.pddl`, `logistics_problem.pddl` | ~10 packages, locations, trucks | 4-5 (load, unload, drive, fly) | ~8 (at, in, connected) | Realistic transport coordination |
| **Blockworld** | `blockworld_domain.pddl`, `blockworld_problem.pddl` | 5-10 blocks, table | 4 (stack, unstack, pick, putdown) | 3 (on, clear, holding) | Simple manipulation |
| **Gripper** | `gripper_domain.pddl`, `gripper_problem.pddl` | Balls, grippers, rooms | 4 (pick, drop, move) | 2 (at, carrying) | Multi-agent coordination |
| **Rooms** | `rooms_domain.pddl`, `rooms_problem.pddl` | Rooms, doors, objects | 3 (move-between, move-within) | 2 (at, connected) | Spatial navigation |

### 5.2 Why These Domains?

- **Complexity Gradient**: Logistics (complex) → Gripper (medium) → Blockworld/Rooms (simple)
- **Diversity**: Different predicate arities and action effects
- **Benchmarking**: Standard planning domain suite (IPC)
- **Scalability**: Test learning with varying domain complexity

---

## 6. Neural Network Architecture (Att_PAM)

### 6.1 Model Definition

```
Input: Sequence of state tokens (60 tokens × 70 dims)
    ↓
[Embedding Layer]
  Linear(70 → 128) + Linear(128 → 256)
    ↓
[4x Attention Block]
  Each block:
    - MultiheadAttention(embed_dim=256, heads=8, dropout=0.1)
    - LayerNorm
    - FeedForward (embed_dim → embed_dim)
    - Dropout (0.1)
    - Residual connections
    ↓
[Output Embedding]
  Linear(256 → 128) + ReLU
  Linear(128 → output_dim) + ReLU
    ↓
Output: Predicted next state (70 dims)
```

### 6.2 Design Rationale

| Component | Rationale |
|-----------|-----------|
| **Multi-head Attention (8 heads)** | Attend to multiple predicate aspects simultaneously |
| **4 Attention Layers** | Stack layers to capture hierarchical state transitions |
| **Layer Normalization** | Stabilize training across different domains |
| **Dropout (0.1)** | Regularization to prevent overfitting on small traces |
| **Residual Connections** | Enable deeper models, improve gradient flow |
| **Feed-forward Networks** | Non-linear transformations after attention |

---

## 7. Evaluation Metrics

### 7.1 Primary Metric: Mean Squared Error (MSE)

```
MSE = (1/N) * Σ(predicted_state - ground_truth_state)²
```

- Measures pixel-wise prediction accuracy in token space
- Lower MSE = better learned action model
- Reported per domain, aggregated across test traces

### 7.2 Learning Curve Analysis

**Graph Generation**:
- X-axis: Training set size (m traces per domain)
- Y-axis: MSE on held-out test traces
- Lines: One per domain (Logistics, Blockworld, Gripper, Rooms)
- Each point: Average of `repetitions` independent runs

**Expected Patterns**:
- MSE decreases as training data increases (learning curve)
- Complex domains (Logistics) may have higher asymptotic error
- Simple domains (Blockworld) may saturate quickly

### 7.3 Storage Format

**eval_results.json**:
```json
{
  "domain_name": {
    "identifier": ["id1", "id2", ...],
    "truth": [ground_truth_mse_1, ...],
    "instance": [model_pred_1, ...],
    "MSE": [final_mse_1, ...],
    "m_values": [m_range] 
  }
}
```

---

## 8. Key Configuration Parameters

| Parameter | Default | Impact |
|-----------|---------|--------|
| `token_size` | 70 | Input/output dimensionality per token |
| `num_tokens` | 60 | Max sequence length (pad shorter traces) |
| `embed_dim` | 256 | Attention layer hidden dimension |
| `num_heads` | 8 | Multi-head attention heads |
| `dropout` | 0.1 | Regularization rate |
| `trace_size` | 5 | Training trace length (actions) |
| `trace_length_eval` | 20 | Evaluation trace length (actions) |
| `m_range` | [2, 10] | Training set size (traces/domain) |
| `repetitions` | 1 | Num independent runs per m value |
| `action_space` | 20 | Max actions (affects token indexing) |
| `pred_offset` | Chain per domain | Predicate index offset (multi-domain) |

---

## 9. Workflow: End-to-End Example

### Step-by-Step Execution

```bash
# 1. Initialize domains and index manager
domains = build_domains()  # Creates 4 Domain_sim objects with chained offsets

# 2. Run training loop
metric_history, summary = await train_and_evaluate(
    domains=domains,
    trace_size=5,                  # 5-step traces
    trace_length_eval=20,           # 20-step evaluation
    m_range_start=2,                # Start with 2 traces/domain
    m_range_end=10,                 # Go up to 10 traces/domain
    repetitions=3                   # 3 independent runs each
)

# 3. For each m value:
#    a. Generate 2×4=8 traces (2 per domain)
#    b. Train model to convergence (delta < ε)
#    c. Evaluate on 4 domains: get MSE
#    d. Record all individual metrics
#    e. Generate learning curve graph

# 4. Output: 
#    - eval_results.json: All metrics
#    - domain_results.json: Summary stats
#    - graphs/[domain]/learning_curve.png: Visualization
```

---

## 10. File Organization

```
new_research/
├── pddlsim_runner.py         # PDDL simulation wrapper & trace generation
├── evaluate.py                # Training pipeline & evaluation loop
├── IndexManager.py            # Predicate-to-token index mapping
├── Model/
│   └── att_only.py           # Att_PAM neural network model
├── Domains/
│   ├── domain_generator.py    # Python domain implementations
│   ├── logistics_domain.py    # Logistics domain class
│   └── __pycache__/
├── pddlDomains/               # PDDL formal specifications
│   ├── logistics_domain.pddl
│   ├── blockworld_domain.pddl
│   ├── gripper_domain.pddl
│   └── rooms_domain.pddl
├── Parameters/                # Trained model checkpoints (.pth)
│   ├── logistics_domain_trained.pth
│   ├── pddl_sim_*.pth
│   └── ...
├── graphs/                    # Generated learning curve visualizations
│   └── logistics/
│       └── learning_curve.png
├── eval_results.json          # Evaluation metrics (all runs)
├── domain_results.json        # Domain-specific summary stats
├── archive/                   # Previous implementations & experiments
│   ├── main.py
│   ├── Network.py
│   └── sequential_domain_training.py
└── README.md
```

---

## 11. Key Research Questions

1. **Learning Efficiency**: How quickly can the attention model learn action effects from traces?
2. **Cross-Domain Transfer**: Does training on one domain help on another?
3. **Scalability**: How does performance degrade with domain complexity?
4. **Convergence**: How does model architecture affect convergence speed?

---

## 12. Dependencies & Requirements

### Core Libraries
- **pddlsim**: PDDL simulation and parsing
- **torch**: Neural network training
- **numpy**: Numerical operations
- **matplotlib**: Graph generation
- **tqdm**: Progress bars

### Python Version
- Python 3.8+

### External Tools
- PDDL planning domains (standard IPC benchmarks)

---

## 13. Known Limitations & Future Work

### Current Limitations
- Random action selection during trace generation (suboptimal)
- Fixed token sequence length may lose long-horizon information
- Single-step prediction focus (not multi-step lookahead)
- No formal validation against ground-truth action models

### Future Directions
- Learn optimal policies instead of random policies
- Hierarchical action abstractions
- Cross-domain transfer learning
- Integration with classical PDDL planners
- Uncertainty quantification in predictions

---

## 14. Usage Notes

### Running Evaluations
```python
# In evaluate.py
async def main():
    domains = build_domains()
    results = await train_and_evaluate(
        domains=domains,
        trace_size=5,
        trace_length_eval=20,
        m_range_start=2,
        m_range_end=10,
        repetitions=3
    )
```

### Adding New Domains
1. Create PDDL files in `pddlDomains/`
2. Define `Domain_sim` object with correct offsets
3. Add to `build_domains()` in `evaluate.py`
4. Chain `pred_offset` and `action_offset`

### Inspecting Traces
```python
# Generate trace from domain
trace = await domain.Generate_trace(trace_size=5)
print(f"Shape: {trace.shape}")  # (num_tokens, token_size)

# First token is action, others are predicates
action_token = trace[0]
predicate_tokens = trace[1:]
```

---

## 15. Research Contributions

- **Neurosymbolic Learning**: Bridges symbolic PDDL planning with deep learning
- **Attention for Action Models**: Novel use of multi-head attention for state prediction
- **Multi-Domain Evaluation**: Systematic evaluation across diverse planning domains
- **Scalability Analysis**: Understanding learning curves across domain complexity

---

**Last Updated**: May 2026  
**Status**: Active Research
