"""
Fusion Generator Module

This module provides classes and functions for generating fusion music by combining
different musical elements, styles, and characteristics.
"""

from .core import FusionGenerator, StyleFusion
from .components import StyleAnalyzer, ElementMixer, FusionParams
from .utils import blend_patterns, harmonize_elements
from .genre_merger import GenreMerger
from .cross_genre_adapter import CrossGenreAdapter
from .fusion_matrix import FusionMatrix

__all__ = [
    'FusionGenerator',
    'StyleFusion',
    'StyleAnalyzer',
    'ElementMixer',
    'FusionParams',
    'blend_patterns',
    'harmonize_elements',
]

