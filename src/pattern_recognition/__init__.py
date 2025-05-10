"""
Pattern Recognition Module

This module provides classes and utilities for musical pattern recognition and analysis.
"""

from .feature_extractor import FeatureExtractor
from .pattern_analyzer import PatternAnalyzer
# Import main classes for easier access
from .pattern_detector import PatternDetector
from .rhythm_matcher import RhythmMatcher
from .sequence_analyzer import SequenceAnalyzer

# Export the classes that should be available directly from the module
__all__ = [
    "PatternDetector",
    "PatternAnalyzer",
    "RhythmMatcher",
    "FeatureExtractor",
    "SequenceAnalyzer",
]
