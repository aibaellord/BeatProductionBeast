"""
Fusion Generator Module

This module provides classes and functions for generating fusion music by combining
different musical elements, styles, and characteristics.
"""

from .core import FusionGenerator, StyleFusion
from .components import StyleAnalyzer, ElementMixer, FusionParams
from .utils import blend_patterns, harmonize_elements

__all__ = [
    'FusionGenerator',
    'StyleFusion',
    'StyleAnalyzer',
    'ElementMixer',
    'FusionParams',
    'blend_patterns',
    'harmonize_elements',
]

