"""
Trace Simulator
Validates extracted predicates via pddlsim simulation
"""

import asyncio
import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pddlsim_runner import Domain_sim
from IndexManager import IndexManager
from config import DOMAINS


class TraceSimulator:
    """
    Simulates extracted predicate states through PDDL domain to validate
    consistency and compute ground truth action/state transitions.
    
    Pipeline:
    1. Input: Extracted predicates from visual network
    2. Simulate via pddlsim: predicates + action → next_state predicates
    3. Compare visual predictions vs simulation ground truth
    4. Compute state/action accuracy
    """
    
    def __init__(self, domain: str):
        """
        Initialize simulator for domain.
        
        Args:
            domain: "hanoi" or "puzzle8"
        """
        self.domain = domain
        self.domain_config = DOMAINS[domain]
        self.index_manager = IndexManager(start_index=0)
        
        # Initialize Domain_sim for simulation
        self.domain_sim = Domain_sim(
            indexManager=self.index_manager,
            DOMAIN_FILE=self.domain_config["domain_file"],
            PROBLEM_FILE=self.domain_config["problem_file"],
            action_space=20,
            action_offset=self._get_action_offset(),
            pred_offset=self._get_pred_offset(),
            token_size=70
        )
        
        self.num_propositions = self.domain_config["article_baseline"]["num_propositions"]
        self.num_actions = self.domain_config["article_baseline"]["num_actions"]
        
        print(f"[trace_simulator] Initialized for domain: {domain}")
        print(f"  Num propositions: {self.num_propositions}")
        print(f"  Num actions: {self.num_actions}")
    
    def _get_action_offset(self) -> int:
        offsets = {"hanoi": 14, "puzzle8": 15}
        return offsets.get(self.domain, 0)
    
    def _get_pred_offset(self) -> int:
        return 20  # After action_space
    
    async def generate_synthetic_trace(
        self,
        trace_length: int = 5
    ) -> Tuple[List[Dict[str, bool]], List[int]]:
        """
        Generate a REAL ground-truth trace by simulating the domain via pddlsim.

        Decodes the 3-slot predicate token encoding produced by
        Domain_sim.get_obj_predicates_Tokens:
            token[indx]   = 1            -> identifier (predicate present)
            token[indx+1] = 1/0          -> truth value
            token[indx+2] = instance #   -> which grounding

        Args:
            trace_length: Number of steps to simulate

        Returns:
            (predicate_sequence, action_sequence) where:
            - predicate_sequence: List of {prop_key: bool} dicts (real truth)
            - action_sequence: List of action indices
        """
        # Generate trace via pddlsim (real domain simulation)
        tokens = await self.domain_sim.Generate_trace(trace_size=trace_length)

        # Build inverse map: identifier index -> predicate name
        index_to_pred = {idx: name for name, idx in self.domain_sim.token_map.items()}

        predicate_sequence = []
        action_sequence = []

        for token_list in tokens:
            if not token_list:
                continue

            # First token is the action one-hot
            action_token = np.asarray(token_list[0])
            action_idx = int(np.argmax(action_token[:self.domain_sim.action_space]))
            action_sequence.append(action_idx)

            # Decode real predicate truth values from remaining tokens
            state_dict = {}
            for token in token_list[1:]:
                token = np.asarray(token)
                # An identifier slot is a predicate-region index set to 1
                for idx, pred_name in index_to_pred.items():
                    if idx + 2 < len(token) and token[idx] == 1:
                        truth = bool(token[idx + 1] >= 0.5)
                        instance = int(round(float(token[idx + 2])))
                        prop_key = f"{pred_name}#{instance}"
                        state_dict[prop_key] = truth
                        break  # one identifier per token

            predicate_sequence.append(state_dict)

        return predicate_sequence, action_sequence
    
    def validate_state_transition(
        self,
        initial_state: Dict[str, bool],
        action: int,
        predicted_next_state: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Check if action applied to initial_state leads to predicted next_state.
        
        Uses PDDL domain logic to determine valid transitions.
        
        Args:
            initial_state: Current predicate state dict
            action: Action index
            predicted_next_state: Next state predicted by visual network
            
        Returns:
            Dict with metrics:
            - valid: bool, whether action is applicable in initial_state
            - match_score: float, similarity between predicted and simulated next state
        """
        # Get action preconditions and effects from domain
        precon, addeff, deleff = self.domain_sim.domain_model.build([action])
        
        # Check if action is applicable (all preconditions satisfied)
        action_valid = True
        for pred_name, is_true in initial_state.items():
            if pred_name in self.domain_sim.token_map:
                pred_idx = self.domain_sim.token_map[pred_name]
                if precon[0, pred_idx].item() > 0.5 and not is_true:
                    action_valid = False
                    break
        
        # If valid, compute expected next state
        if action_valid:
            expected_next = initial_state.copy()
            for pred_name in initial_state:
                if pred_name in self.domain_sim.token_map:
                    pred_idx = self.domain_sim.token_map[pred_name]
                    if deleff[0, pred_idx].item() > 0.5:
                        expected_next[pred_name] = False
                    if addeff[0, pred_idx].item() > 0.5:
                        expected_next[pred_name] = True
            
            # Compare with predicted
            match_count = sum(
                1 for p in initial_state
                if initial_state[p] == predicted_next_state.get(p, False)
            )
            match_score = match_count / len(initial_state)
        else:
            match_score = 0.0
        
        return {
            "action_applicable": action_valid,
            "state_match_score": match_score,
        }
    
    def compute_state_accuracy(
        self,
        predicted_states: List[Dict[str, bool]],
        ground_truth_states: List[Dict[str, bool]]
    ) -> float:
        """
        Compute state prediction accuracy.
        
        Args:
            predicted_states: Visual network predictions
            ground_truth_states: Ground truth from simulation
            
        Returns:
            Fraction of correctly predicted propositions
        """
        if not predicted_states:
            return 0.0
        
        correct = 0
        total = 0
        
        for pred_state, gt_state in zip(predicted_states, ground_truth_states):
            for pred_name in gt_state:
                if pred_name in pred_state:
                    if pred_state[pred_name] == gt_state[pred_name]:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0
    
    def compute_action_accuracy(
        self,
        predicted_actions: List[int],
        ground_truth_actions: List[int]
    ) -> float:
        """
        Compute action prediction accuracy.
        
        Args:
            predicted_actions: Extracted action indices
            ground_truth_actions: Annotated action indices
            
        Returns:
            Fraction of correctly predicted actions
        """
        if not predicted_actions:
            return 0.0
        
        correct = sum(
            1 for pred, gt in zip(predicted_actions, ground_truth_actions)
            if pred == gt
        )
        
        return correct / len(predicted_actions)
    
    def compute_trace_metrics(
        self,
        predicted_sequence: List[Dict[str, bool]],
        ground_truth_sequence: List[Dict[str, bool]],
        action_sequence: List[int]
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for a trace.
        
        Args:
            predicted_sequence: Visual predictions
            ground_truth_sequence: Simulation ground truth
            action_sequence: Executed actions
            
        Returns:
            Dict with metrics:
            - state_accuracy: Predicate accuracy
            - action_validity_rate: Fraction of applicable actions
            - trace_deviation: Average state divergence over trace
        """
        state_acc = self.compute_state_accuracy(predicted_sequence, ground_truth_sequence)
        
        # Action validity
        valid_actions = 0
        for i, action in enumerate(action_sequence):
            if i < len(predicted_sequence) - 1:
                metrics = self.validate_state_transition(
                    predicted_sequence[i],
                    action,
                    predicted_sequence[i + 1]
                )
                if metrics["action_applicable"]:
                    valid_actions += 1
        
        action_validity = valid_actions / len(action_sequence) if action_sequence else 0.0
        
        # State divergence (how much predicted diverges from ground truth)
        divergence = 1.0 - state_acc
        
        return {
            "state_accuracy": state_acc,
            "action_validity_rate": action_validity,
            "trace_deviation": divergence,
        }
