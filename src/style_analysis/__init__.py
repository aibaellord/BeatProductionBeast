"""
Style Analysis Module

This module provides tools for analyzing musical styles and genres,
identifying key characteristics, and extracting style-specific features.
"""

from .analyzer import StyleAnalyzer
from .classifier import GenreClassifier, StyleClassifier
from .features import StyleFeatureExtractor, GenreFeatures
from .comparison import StyleComparator
from .profiles import StyleProfile, GenreProfile

__all__ = [
    'StyleAnalyzer',
    'GenreClassifier',
    'StyleClassifier',
    'StyleFeatureExtractor',
    'GenreFeatures',
    'StyleComparator',
    'StyleProfile',
    'GenreProfile',
]

