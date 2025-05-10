"""
Style Analysis module for BeatProductionBeast.

This module provides functionality for analyzing music styles, extracting
patterns, and generating reports on musical characteristics.
"""

__all__ = []  # Add module exports as they are implemented

"""
Style Analysis Module

This module provides tools for analyzing musical styles and genres,
identifying key characteristics, and extracting style-specific features.
"""

from .analyzer import StyleAnalyzer
from .classifier import GenreClassifier, StyleClassifier
from .comparison import StyleComparator
from .features import GenreFeatures, StyleFeatureExtractor
from .profiles import GenreProfile, StyleProfile

__all__ = [
    "StyleAnalyzer",
    "GenreClassifier",
    "StyleClassifier",
    "StyleFeatureExtractor",
    "GenreFeatures",
    "StyleComparator",
    "StyleProfile",
    "GenreProfile",
]
