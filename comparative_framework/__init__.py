"""
Comparative Framework Package
ROSAME Visual Network → Token Translation → Model Inference → Comparative Analysis
"""

from .config import DOMAINS, TRAINING_CONFIG, MODEL_CONFIG, ROSAME_CONFIG
from .visual_extraction import VisualExtractor, get_visual_extractor
from .token_translator import TokenTranslator
from .trace_simulator import TraceSimulator
from .comparative_tester import ComparativeTest

__all__ = [
    "DOMAINS",
    "TRAINING_CONFIG", 
    "MODEL_CONFIG",
    "ROSAME_CONFIG",
    "VisualExtractor",
    "get_visual_extractor",
    "TokenTranslator",
    "TraceSimulator",
    "ComparativeTest",
]
