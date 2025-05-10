"""
Fusion Generator Module

This module provides classes and functions for generating fusion music by combining
different musical elements, styles, and characteristics.
"""

from .components import ElementMixer, FusionParams, StyleAnalyzer
from .core import FusionGenerator, StyleFusion
from .cross_genre_adapter import CrossGenreAdapter
from .fusion_matrix import FusionMatrix
from .genre_merger import GenreMerger
from .utils import blend_patterns, harmonize_elements

__all__ = [
    "FusionGenerator",
    "StyleFusion",
    "StyleAnalyzer",
    "ElementMixer",
    "FusionParams",
    "blend_patterns",
    "harmonize_elements",
]
