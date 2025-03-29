"""
Pattern Recognition Module

This module provides classes and utilities for musical pattern recognition and analysis.
"""

# Import main classes for easier access
from .pattern_detector import PatternDetector
from .pattern_analyzer import PatternAnalyzer
from .rhythm_matcher import RhythmMatcher
from .feature_extractor import FeatureExtractor
from .sequence_analyzer import SequenceAnalyzer

# Export the classes that should be available directly from the module
__all__ = [
    'PatternDetector',
    'PatternAnalyzer',
    'RhythmMatcher',
    'FeatureExtractor',
    'SequenceAnalyzer',
]

