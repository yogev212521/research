"""
Token Translator
Maps extracted predicates → IndexManager tokens → model-ready format
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from IndexManager import IndexManager
from pddlsim_runner import Domain_sim
from config import DOMAINS, MODEL_CONFIG


class TokenTranslator:
    """
    Bridges extracted predicates to IndexManager token encoding.
    
    Pipeline:
    1. Input: Dict[predicate_name] → bool
    2. Output: Token tensor (60 tokens × 70 dims)
    
    Token encoding (per Model_Config):
    - token_size: 70 dimensions
    - num_tokens: 60 tokens per trace
    - action_offset: 0-19 (action one-hot)
    - pred_offset: 20+ (predicate tokens)
    """
    
    def __init__(self, domain: str):
        """
        Initialize translator with domain-specific mappings.
        
        Args:
            domain: "hanoi" or "puzzle8"
        """
        self.domain = domain
        self.domain_config = DOMAINS[domain]
        
        # Set parameters first (before they're used)
        self.num_tokens = MODEL_CONFIG["num_tokens"]
        self.token_size = MODEL_CONFIG["token_size"]
        self.action_space = MODEL_CONFIG["action_space"]
        
        # Initialize IndexManager
        self.index_manager = IndexManager(start_index=0)
        
        # Initialize Domain_sim to get token mappings
        self.domain_sim = Domain_sim(
            indexManager=self.index_manager,
            DOMAIN_FILE=self.domain_config["domain_file"],
            PROBLEM_FILE=self.domain_config["problem_file"],
            action_space=self.action_space,
            action_offset=self._get_action_offset(),
            pred_offset=self._get_pred_offset(),
            token_size=self.token_size
        )
        
        print(f"[token_translator] Initialized for domain: {domain}")
        print(f"  Token map: {len(self.domain_sim.token_map)} predicates")
        print(f"  Action space: {self.action_space}")
        print(f"  Token size: {self.token_size}")
    
    def _get_action_offset(self) -> int:
        """Get action offset for this domain (from evaluate.py)."""
        offsets = {
            "hanoi": 14,  # From evaluate.py
            "puzzle8": 15,
        }
        return offsets.get(self.domain, 0)
    
    def _get_pred_offset(self) -> int:
        """Get predicate offset for this domain."""
        # Start after action space
        return self.action_space
    
    def predicates_to_action_token(
        self, 
        action_index: int
    ) -> np.ndarray:
        """
        Convert action index to one-hot token.
        
        Args:
            action_index: Integer in [0, action_space)
            
        Returns:
            Token array (token_size,)
        """
        token = np.zeros(self.token_size, dtype=np.float32)
        if 0 <= action_index < self.action_space:
            token[action_index] = 1.0
        return token
    
    def predicates_to_state_tokens(
        self,
        predicates: Dict[str, bool],
        predicate_names: List[str]
    ) -> List[np.ndarray]:
        """
        Convert predicate dict to state tokens.
        
        Args:
            predicates: Dict mapping predicate names to boolean values
            predicate_names: Canonical list of predicate names in order
            
        Returns:
            List of predicate tokens for relevant args
        """
        tokens = []
        
        for pred_name in predicate_names:
            token = np.zeros(self.token_size, dtype=np.float32)
            
            if pred_name in self.domain_sim.token_map:
                pred_token_idx = self.domain_sim.token_map[pred_name]
                
                # One-hot encode predicate with its truth value
                # Index: predicate index in token dimension
                # Value: 1.0 if true, -1.0 if false (or 0 for unknown)
                if pred_name in predicates:
                    token[pred_token_idx] = 1.0 if predicates[pred_name] else -1.0
                else:
                    token[pred_token_idx] = 0.0  # Unknown
            
            tokens.append(token)
        
        return tokens
    
    def state_dict_to_token_sequence(
        self,
        state_dict: Dict[str, bool],
        predicate_names: List[str]
    ) -> np.ndarray:
        """
        Convert predicate state dict to a single token sequence.
        
        Args:
            state_dict: Predicate name → boolean
            predicate_names: Ordered list of all predicates
            
        Returns:
            Array of shape (num_tokens, token_size) padded with zeros
        """
        tokens = []
        
        # Create predicate tokens
        state_tokens = self.predicates_to_state_tokens(state_dict, predicate_names)
        tokens.extend(state_tokens[:self.num_tokens])
        
        # Pad to fixed size
        while len(tokens) < self.num_tokens:
            tokens.append(np.zeros(self.token_size, dtype=np.float32))
        
        return np.array(tokens[:self.num_tokens], dtype=np.float32)
    
    def trace_to_token_pairs(
        self,
        predicate_sequence: List[Dict[str, bool]],
        action_sequence: List[int],
        predicate_names: List[str]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert predicate trace to (state_token, action_token) pairs.
        
        Args:
            predicate_sequence: List of state dicts over time
            action_sequence: List of action indices [0, action_space)
            predicate_names: Ordered canonical predicate names
            
        Returns:
            List of (state_token_seq, action_token) pairs
        """
        pairs = []
        
        # Ensure sequences have same length
        min_len = min(len(predicate_sequence), len(action_sequence))
        
        for t in range(min_len):
            state_tokens = self.state_dict_to_token_sequence(
                predicate_sequence[t], 
                predicate_names
            )
            action_token = self.predicates_to_action_token(action_sequence[t])
            
            pairs.append((state_tokens, action_token))
        
        return pairs
    
    def batch_predicate_sequences(
        self,
        predicate_sequences: List[List[Dict[str, bool]]],
        action_sequences: List[List[int]],
        predicate_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch multiple predicate sequences into tensors for model input.
        
        Args:
            predicate_sequences: List of traces (each trace is list of state dicts)
            action_sequences: List of traces (each trace is list of action indices)
            predicate_names: Ordered predicate names
            
        Returns:
            (state_batch, action_batch) of shape:
            - state_batch: (batch, time, num_tokens, token_size)
            - action_batch: (batch, time, token_size)
        """
        state_batch = []
        action_batch = []
        
        for preds_trace, actions_trace in zip(predicate_sequences, action_sequences):
            pairs = self.trace_to_token_pairs(preds_trace, actions_trace, predicate_names)
            
            if pairs:
                states, actions = zip(*pairs)
                state_batch.append(np.array(states))  # (T, num_tokens, token_size)
                action_batch.append(np.array(actions))  # (T, token_size)
        
        # Stack and convert to tensors
        if state_batch:
            state_tensor = torch.from_numpy(np.array(state_batch)).float()
            action_tensor = torch.from_numpy(np.array(action_batch)).float()
        else:
            state_tensor = torch.zeros((0, 0, self.num_tokens, self.token_size))
            action_tensor = torch.zeros((0, 0, self.token_size))
        
        return state_tensor, action_tensor
    
    def get_predicate_names(self) -> List[str]:
        """Get canonical ordered predicate names."""
        return sorted(self.domain_sim.token_map.keys())
    
    def get_action_count(self) -> int:
        """Get total number of grounded actions."""
        return len(self.domain_sim.domain.actions_section._items)
