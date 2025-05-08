"""
Audio Engine Module for BeatProductionBeast

This module provides audio processing, sound generation, and mixing functionality.
"""

from .core import AudioProcessor
from .sound_generator import SoundGenerator
from .mixer_interface import MixerInterface
from .frequency_modulator import FrequencyModulator
from .waveform_analyzer import WaveformAnalyzer
from .audio_effect import AudioEffect

__all__ = [
    'AudioProcessor',
    'SoundGenerator',
    'MixerInterface',
    'FrequencyModulator',
    'WaveformAnalyzer',
]

