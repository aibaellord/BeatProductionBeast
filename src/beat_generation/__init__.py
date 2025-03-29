"""
Beat Generation Module

This module handles the generation of musical beats and patterns.
"""

# Import main classes from submodules
# Adjust these imports based on your actual implementation
from .beat_generator import BeatGenerator
from .pattern_creator import PatternCreator
from .rhythm_engine import RhythmEngine
from .sequence_builder import SequenceBuilder

# Define what's available when importing with *
__all__ = [
    'BeatGenerator',
    'PatternCreator',
    'RhythmEngine',
    'SequenceBuilder',
]

