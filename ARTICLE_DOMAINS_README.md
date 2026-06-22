# Article Domains Integration

## Summary

Successfully extracted domain specifications from the research paper (2604.19043v2: "Learning Lifted Action Models from Unsupervised Visual Traces") and integrated two new classical planning domains into the neurosymbolic learning framework.

## Paper Information

**Title:** Learning Lifted Action Models from Unsupervised Visual Traces

**Authors:** Kai Xi, Stephen Gould, Sylvie Thiébaux

**Publication:** arXiv:2604.19043v2 (2026)

**Key Contributions:**
- Framework for learning lifted action models from visual traces without action supervision
- Deep learning component for state and action prediction
- MILP (Mixed-Integer Linear Programming) module for logical consistency
- Evaluation across 6 classical planning domains

## Domains from the Article

The paper evaluates the following domains:

### Existing Domains (Already in System)
1. **Blocksworld (MNIST grid)** - 36 propositions, 50 actions, 5 blocks
2. **Blocksworld (Synthesized)** - PDDLGym-rendered version
3. **Gripper** - 28 propositions, 50 actions, 6 balls, 2 grippers, 2 rooms
4. **Logistics** - 72 propositions, 196 actions, 6 packages, 2 trucks, 2 airplanes

### New Domains (Added)
5. **Hanoi (Tower of Hanoi)** - 55 propositions, 120 actions, 4 discs, 3 pegs
6. **8-Puzzle** - 153 propositions, 576 actions, 8 tiles on 3×3 board

## Files Added/Modified

### PDDL Domain Files
- `pddlDomains/hanoi_domain.pddl` - Tower of Hanoi domain specification
- `pddlDomains/hanoi_problem.pddl` - 4-disc initial/goal problem instance
- `pddlDomains/puzzle8_domain.pddl` - 8-puzzle domain specification
- `pddlDomains/puzzle8_problem.pddl` - Initial/goal state problem instance

### Evaluation Scripts
- `evaluate.py` - Updated with Hanoi and 8-puzzle integration (now supports 6 domains)
- `evaluate_article_domains.py` - New script focusing exclusively on article domains

### Supporting Files
- `pdf_reader.py` - PDF text extraction utility (uses PyPDF2)

## Domain Specifications

### Hanoi (Tower of Hanoi)
- **Objects:** 4 discs (d1, d2, d3, d4), 3 pegs (peg1, peg2, peg3)
- **Predicates:**
  - `on(disc, peg)` - disc is on a peg
  - `clear(peg)` - peg has no discs on top
  - `smaller(disc, disc)` - size ordering constraint
- **Actions:**
  - `move(disc, peg, peg)` - move disc from source peg to destination peg
  - **Preconditions:** Source peg contains disc, destination peg is clear, disc respects size constraint
  - **Effects:** Update disc location and peg clear status
- **Complexity:** 5 steps (from paper), 120 action ground instances

### 8-Puzzle
- **Objects:** 8 tiles (t1-t8), 9 positions (p1-p9 in a 3×3 grid)
- **Predicates:**
  - `at(tile, position)` - tile is at a position
  - `blank(position)` - position is empty
  - `adjacent(position, position)` - grid adjacency
- **Actions:**
  - `move(tile, position, position)` - slide tile into blank space
  - **Preconditions:** Tile at source position, blank at destination, positions adjacent
  - **Effects:** Update tile and blank locations
- **Complexity:** 5 steps (from paper), 576 action ground instances
- **Grid Structure:** 
  ```
  p1 p2 p3
  p4 p5 p6
  p7 p8 p9
  ```

## Integration with Existing System

### Token Encoding
Both domains follow the existing token encoding scheme:
- 60 tokens per sequence
- 70 dimensions per token
- Action token (one-hot) + predicate tokens (3-dim each) + padding

### Index Manager Integration
New domains are chained with offset system:
- Hanoi: pred_offset from Rooms domain, action_offset = 14
- 8-Puzzle: pred_offset from Hanoi, action_offset = 15

### Multi-Domain Evaluation
- `evaluate.py` builds all 6 domains with automatic offset chaining
- `evaluate_article_domains.py` focuses on just Hanoi and 8-puzzle
- Both can scale experiments with variable training traces (m-range)

## Usage

### Run Full System (All 6 Domains)
```bash
python evaluate.py
```

### Run Article Domains Only
```bash
python evaluate_article_domains.py
```

### Extract Text from Article
```bash
python pdf_reader.py "2604.19043v2 (1).pdf"
```

## Experimental Results from Paper

Table 1 shows baseline results on these domains:

| Domain | Props | Actions | Steps | MILP Agree | State Acc | Action Acc |
|--------|-------|---------|-------|-----------|-----------|------------|
| Blocksworld (MNIST) | 36 | 50 | 3 | 0.977 | 97.81% | 85.33% |
| Gripper | 28 | 50 | 9 | 0.978 | 100% | 100% |
| Logistics | 72 | 196 | 9 | 0.983 | 99.89% | 99.56% |
| Blocksworld (Synth) | 36 | 50 | 3 | 0.976 | 99.29% | 88.67% |
| **Hanoi** | **55** | **120** | **5** | **0.940** | **98.55%** | **81.40%** |
| **8-Puzzle** | **153** | **576** | **5** | **0.985** | **99.77%** | **92.60%** |

## Architecture

The system pipeline:
```
PDDL Domains → LocalSimulator → Trace Generation
     ↓
Token Encoding (60 tokens × 70 dims)
     ↓
Att_PAM Neural Network (4 attention layers, 8 heads)
     ↓
State/Action Prediction
     ↓
Evaluation Metrics (MSE, Agreement, Accuracy)
```

## Paper Methodology

The paper's framework includes:
1. **Deep Learning Component:** Joint state and action prediction from visual traces
2. **ROSAME Integration:** Lifts learned action models to symbolic level
3. **MILP Correction:** Enforces logical consistency over trajectory subsets
4. **Iterative Training:** MILP pseudo-labels guide neural network refinement

## Next Steps

Potential extensions:
1. Implement visual representations (MNIST/synthesized images) for Hanoi and 8-puzzle
2. Add MILP constraint enforcement to the training loop
3. Experiment with longer trace lengths
4. Integrate hierarchical planning capabilities
5. Test generalization to instances with different object counts

## Resources

- **Paper:** https://arxiv.org/abs/2604.19043v2
- **Code Reference:** GitHub repository link from paper
- **Related Work:** Latplan, ROSAME, FAMA, and other neuro-symbolic learning approaches

---

**Date Created:** May 23, 2026  
**System:** Neurosymbolic Learning for Action Model Discovery  
**Framework Version:** Extended with article domains
