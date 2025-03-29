"""
Audio Engine Module for BeatProductionBeast

This module provides audio processing, sound generation, and mixing functionality.
"""

from .audio_processor import AudioProcessor
from .sound_generator import SoundGenerator
from .mixer_interface import MixerInterface

__all__ = [
    'AudioProcessor',
    'SoundGenerator',
    'MixerInterface',
]

