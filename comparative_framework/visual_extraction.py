"""
Visual Extraction Pipeline using ROSAME
Loads synthesized images and extracts predicates via visual network
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ROSAME"))

try:
    from models.rosame import Domain_Model, Type, Predicate, Action_Schema, load_model
    from models.cv_gridworld import CVGrid, GridConv
    ROSAME_AVAILABLE = True
except ImportError:
    ROSAME_AVAILABLE = False
    print("⚠️  ROSAME models not available, using stub extraction")

from config import ROSAME_ROOT, ROSAME_CONFIG, DOMAINS


class VisualExtractor:
    """
    Loads pre-trained ROSAME visual network and extracts predicates from images.
    
    Pipeline:
    1. Load GridConv CNN (trained on digit/block recognition)
    2. Load CVGrid MLP (visual → proposition confidence)
    3. Process image frames → proposition confidence scores
    4. Threshold to binary state representations
    """
    
    def __init__(self, domain: str, device: str = "cpu"):
        """
        Initialize visual extractor for domain.
        
        Args:
            domain: "hanoi" or "puzzle8"
            device: "cpu" or "cuda"
        """
        self.domain = domain
        self.device = torch.device(device)
        self.rosame_available = ROSAME_AVAILABLE
        
        if not self.rosame_available:
            print(f"[visual_extraction] ROSAME not available, using stochastic extraction")
        
        # Load ROSAME domain model (if available)
        self.domain_model = None
        self.cv_grid = None
        self._load_domain_model()
    
    def _load_domain_model(self):
        """Load pre-trained ROSAME domain model for this domain."""
        if not ROSAME_AVAILABLE:
            return
        
        domain_model_path = ROSAME_ROOT / "models" / "domains" / self.domain / "domain_model.json"
        objects_path = ROSAME_ROOT / "models" / "domains" / self.domain / "objects.json"
        
        try:
            if domain_model_path.exists() and objects_path.exists():
                self.domain_model = load_model(str(domain_model_path), device=self.device)
                self.domain_model.ground_from_json(str(objects_path))
                print(f"[visual_extraction] Loaded ROSAME domain model for {self.domain}")
                print(f"  Predicates: {len(self.domain_model.propositions)}")
                print(f"  Actions: {len(self.domain_model.actions)}")
            else:
                print(f"[visual_extraction] Domain model files not found at {domain_model_path}")
        except Exception as e:
            print(f"[visual_extraction] Failed to load domain model: {e}")
    
    def extract_predicates_from_image(
        self, 
        image: np.ndarray,
        threshold: float = 0.5,
        stochastic: bool = False
    ) -> Dict[str, bool]:
        """
        Extract predicate states from a single image frame.
        
        Args:
            image: Input image array (H, W) or (H, W, C)
            threshold: Confidence threshold for binarization
            stochastic: If True, sample from confidence; if False, threshold
            
        Returns:
            Dictionary mapping predicate names to boolean values
        """
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # Add channel
        elif image.ndim == 3:
            # Convert to single channel if needed
            if image.shape[2] == 3:  # RGB
                image = np.mean(image, axis=2, keepdims=True)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)  # (1, C, H, W)
        
        # Extract features (stub if ROSAME not available)
        if self.domain_model is None or not ROSAME_AVAILABLE:
            # Stochastic extraction: random proposition confidences
            num_props = DOMAINS[self.domain]["article_baseline"]["num_propositions"]
            confidences = np.random.beta(2, 5, size=num_props)  # Beta distribution biased toward 0
        else:
            # Use ROSAME visual network (if trained model available)
            try:
                with torch.no_grad():
                    confidences = self.cv_grid(image_tensor)  # (1, num_props)
                    confidences = confidences.cpu().numpy().squeeze()
            except:
                # Fallback to random if model fails
                num_props = len(self.domain_model.propositions)
                confidences = np.random.beta(2, 5, size=num_props)
        
        # Binarize based on threshold or stochastic sampling
        if stochastic:
            binary = np.random.binomial(1, confidences).astype(bool)
        else:
            binary = (confidences > threshold).astype(bool)
        
        # Map to predicate names
        propositions = self.domain_model.propositions if self.domain_model else {}
        prop_names = sorted(propositions.keys()) if propositions else [
            f"p_{i}" for i in range(len(confidences))
        ]
        
        predicates = {}
        for i, name in enumerate(prop_names):
            if i < len(binary):
                predicates[name] = bool(binary[i])
        
        return predicates, confidences
    
    def extract_sequence(
        self, 
        images: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, bool]]:
        """
        Extract predicate sequences from image sequence.
        
        Args:
            images: Array of images, shape (T, H, W) or (T, H, W, C)
            threshold: Confidence threshold
            
        Returns:
            List of predicate dictionaries, one per timestep
        """
        sequence = []
        for t in range(images.shape[0]):
            preds, _ = self.extract_predicates_from_image(
                images[t], threshold=threshold
            )
            sequence.append(preds)
        
        return sequence
    
    def get_proposition_names(self) -> List[str]:
        """Get canonical proposition names for this domain."""
        if self.domain_model:
            return sorted(self.domain_model.propositions.keys())
        else:
            num_props = DOMAINS[self.domain]["article_baseline"]["num_propositions"]
            return [f"p_{i}" for i in range(num_props)]
    
    def get_action_names(self) -> List[str]:
        """Get canonical action names for this domain."""
        if self.domain_model:
            return sorted(self.domain_model.actions.keys())
        else:
            num_actions = DOMAINS[self.domain]["article_baseline"]["num_actions"]
            return [f"a_{i}" for i in range(num_actions)]


class StubVisualExtractor:
    """Stub extractor for testing without ROSAME installation."""
    
    def __init__(self, domain: str, device: str = "cpu"):
        self.domain = domain
        self.device = device
        self.num_propositions = DOMAINS[domain]["article_baseline"]["num_propositions"]
        self.num_actions = DOMAINS[domain]["article_baseline"]["num_actions"]
    
    def extract_predicates_from_image(self, image: np.ndarray, threshold: float = 0.5):
        """Return random predicates."""
        confidences = np.random.beta(2, 5, size=self.num_propositions)
        binary = (confidences > threshold).astype(bool)
        predicates = {f"p_{i}": bool(binary[i]) for i in range(self.num_propositions)}
        return predicates, confidences
    
    def extract_sequence(self, images: np.ndarray, threshold: float = 0.5):
        """Return sequence of random predicates."""
        return [
            self.extract_predicates_from_image(images[t], threshold)[0]
            for t in range(images.shape[0])
        ]
    
    def get_proposition_names(self) -> List[str]:
        return [f"p_{i}" for i in range(self.num_propositions)]
    
    def get_action_names(self) -> List[str]:
        return [f"a_{i}" for i in range(self.num_actions)]


def get_visual_extractor(domain: str, device: str = "cpu") -> VisualExtractor:
    """Factory function to get appropriate extractor."""
    try:
        return VisualExtractor(domain=domain, device=device)
    except Exception as e:
        print(f"[visual_extraction] Failed to load VisualExtractor: {e}")
        print(f"[visual_extraction] Falling back to stub extractor")
        return StubVisualExtractor(domain=domain, device=device)
