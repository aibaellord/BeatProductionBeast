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
from .beat_maker import BeatMaker
from .drum_sequencer import DrumSequencer
from .loop_generator import LoopGenerator
from .pattern_library import PatternLibrary

# Define what's available when importing with *
__all__ = [
    'BeatGenerator',
    'PatternCreator',
    'RhythmEngine',
    'SequenceBuilder',
]

