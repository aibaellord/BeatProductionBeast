"""
Harmonic Enhancement Module

This module provides functionality for analyzing and enhancing harmonic content
in audio and MIDI data, improving chord progressions, and applying music theory
principles to create more harmonically rich compositions.
"""

# Import core classes for convenient access
try:
    from .harmonic_analyzer import HarmonicAnalyzer
    from .chord_enhancer import ChordEnhancer
    from .scale_detector import ScaleDetector
    from .progression_generator import ProgressionGenerator
    from .harmonic_filter import HarmonicFilter
    from .pitch_corrector import PitchCorrector
    from .tonal_adjuster import TonalAdjuster
except ImportError as e:
    import warnings
    warnings.warn(f"Some harmonic_enhancement components could not be imported: {e}")

# Define the public API
__all__ = [
    'HarmonicAnalyzer',
    'ChordEnhancer',
    'ScaleDetector',
    'ProgressionGenerator',
    'HarmonicFilter',
    'PitchCorrector',
    'TonalAdjuster',
]

# Module metadata
__version__ = '0.1.0'
__author__ = 'BeatProductionBeast Team'

