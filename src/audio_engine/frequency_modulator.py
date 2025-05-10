"""
Frequency Modulation Module

This module provides frequency modulation capabilities for consciousness enhancement,
including binaural beats generation and brainwave entrainment with advanced sacred geometry principles.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

from src.utils.sacred_geometry_core import SacredGeometryCore

from .core import AudioProcessor

logger = logging.getLogger(__name__)


@dataclass
class ModulationParams:
    base_frequency: float
    consciousness_level: int
    modulation_depth: float
    phi_ratio: float
    harmonic_series: List[float]


class BrainwaveRange(Enum):
    """Brainwave frequency ranges associated with different mental states."""

    DELTA = (0.5, 4)  # Deep sleep, healing
    THETA = (4, 8)  # Meditation, creativity
    ALPHA = (8, 14)  # Relaxation, calmness
    BETA = (14, 30)  # Focus, alertness
    GAMMA = (30, 100)  # Higher cognition, insight
    LAMBDA = (100, 200)  # Advanced states of consciousness
    EPSILON = (200, 400)  # Transcendental states


class SolfeggioFrequency(Enum):
    """Ancient Solfeggio frequencies with spiritual significance."""

    UT = 396  # Liberating guilt and fear
    RE = 417  # Undoing situations and facilitating change
    MI = 528  # Transformation and miracles (DNA repair)
    FA = 639  # Connecting/relationships
    SOL = 741  # Awakening intuition
    LA = 852  # Returning to spiritual order
    SI = 963  # Awakening and returning to oneness


class SacredGeometryRatio(Enum):
    """Sacred geometry ratios for frequency relationships."""

    PHI = 1.618033988749895  # Golden ratio
    PI = math.pi  # Circle's circumference to diameter
    SQRT2 = math.sqrt(2)  # Diagonal of a square
    SQRT3 = math.sqrt(3)  # Altitude of an equilateral triangle
    SQRT5 = math.sqrt(5)  # Diagonal of a golden rectangle
    FIBONACCI_RATIO = 0.618033988749895  # 1/φ
    SCHUMANN = 7.83  # Earth's resonant frequency


class FrequencyModulator:
    """
    Advanced frequency modulation system that applies consciousness-based
    transformations and sacred geometry principles to audio frequencies.
    """

    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self._initialize_modulation_params()

    def _initialize_modulation_params(self):
        """Initialize frequency modulation parameters"""
        self.params = {
            "base_frequency": 432,  # Hz
            "consciousness_multiplier": 12,  # Hz per level
            "modulation_depths": {"consciousness": 0.3, "phi": 0.2, "harmonic": 0.5},
            "frequency_bands": {
                "sub_bass": (20, 60),
                "bass": (60, 250),
                "low_mid": (250, 500),
                "mid": (500, 2000),
                "high_mid": (2000, 4000),
                "high": (4000, 20000),
            },
        }

    def apply_consciousness_modulation(
        self, audio_data: np.ndarray, consciousness_level: int
    ) -> np.ndarray:
        """
        Apply consciousness-based frequency modulation to audio.

        Args:
            audio_data: Input audio signal
            consciousness_level: Target consciousness level (1-13)

        Returns:
            Frequency-modulated audio signal
        """
        try:
            # Calculate modulation parameters
            mod_params = self._calculate_modulation_params(consciousness_level)

            # Apply sacred geometry frequency modulation
            sacred_modulated = self._apply_sacred_geometry_modulation(
                audio_data, mod_params
            )

            # Apply consciousness band modulation
            consciousness_modulated = self._apply_consciousness_bands(
                sacred_modulated, mod_params
            )

            # Apply harmonic series enhancement
            harmonic_enhanced = self._apply_harmonic_enhancement(
                consciousness_modulated, mod_params
            )

            return harmonic_enhanced

        except Exception as e:
            logger.error(f"Error in frequency modulation: {str(e)}")
            raise

    def _calculate_modulation_params(
        self, consciousness_level: int
    ) -> ModulationParams:
        """Calculate modulation parameters based on consciousness level"""
        # Calculate base frequency shift
        base_freq = (
            self.params["base_frequency"]
            + consciousness_level * self.params["consciousness_multiplier"]
        )

        # Calculate modulation depth based on consciousness
        mod_depth = self.params["modulation_depths"]["consciousness"] * (
            consciousness_level / 13.0
        )

        # Generate harmonic series
        harmonics = [base_freq * (self.phi**n) for n in range(consciousness_level)]

        return ModulationParams(
            base_frequency=base_freq,
            consciousness_level=consciousness_level,
            modulation_depth=mod_depth,
            phi_ratio=self.phi,
            harmonic_series=harmonics,
        )

    def _apply_sacred_geometry_modulation(
        self, audio: np.ndarray, params: ModulationParams
    ) -> np.ndarray:
        """Apply sacred geometry-based frequency modulation"""
        # Convert to frequency domain
        spectrum = fft(audio)
        frequencies = np.fft.fftfreq(len(audio))

        # Create phi-based modulation function
        mod_function = np.exp(1j * params.phi_ratio * frequencies * 2 * np.pi)

        # Apply modulation
        modulated_spectrum = spectrum * mod_function

        # Convert back to time domain
        modulated = np.real(ifft(modulated_spectrum))

        return modulated

    def _apply_consciousness_bands(
        self, audio: np.ndarray, params: ModulationParams
    ) -> np.ndarray:
        """Apply consciousness-level specific frequency band modulation"""
        modulated = np.zeros_like(audio)

        for band_name, (low_freq, high_freq) in self.params["frequency_bands"].items():
            # Design bandpass filter
            band = self._create_band_filter(low_freq, high_freq, len(audio))

            # Extract band
            band_data = self._apply_filter(audio, band)

            # Calculate band-specific modulation
            mod_factor = self._calculate_band_modulation(band_name, params)

            # Apply modulation and add to output
            modulated += band_data * mod_factor

        return modulated

    def _apply_harmonic_enhancement(
        self, audio: np.ndarray, params: ModulationParams
    ) -> np.ndarray:
        """Apply harmonic series enhancement"""
        enhanced = np.zeros_like(audio)

        # Apply each harmonic frequency
        for harmonic in params.harmonic_series:
            # Create harmonic modulation
            mod_signal = np.sin(
                2 * np.pi * harmonic * np.arange(len(audio)) / len(audio)
            )

            # Apply modulation with decreasing intensity for higher harmonics
            intensity = 1.0 / (harmonic / params.base_frequency)
            enhanced += audio * mod_signal * intensity * params.modulation_depth

        # Blend with original
        blend_factor = 0.7  # 70% original, 30% enhanced
        return blend_factor * audio + (1 - blend_factor) * enhanced

    def _create_band_filter(
        self, low_freq: float, high_freq: float, length: int
    ) -> np.ndarray:
        """Create bandpass filter"""
        freqs = np.fft.fftfreq(length)

        # Create frequency mask
        mask = np.logical_and(
            abs(freqs * length) >= low_freq, abs(freqs * length) <= high_freq
        )

        return mask.astype(float)

    def _apply_filter(self, audio: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
        """Apply frequency domain filter"""
        spectrum = fft(audio)
        filtered_spectrum = spectrum * filter_mask
        return np.real(ifft(filtered_spectrum))

    def _calculate_band_modulation(
        self, band_name: str, params: ModulationParams
    ) -> float:
        """Calculate band-specific modulation factor"""
        # Base modulation factors for each band
        base_factors = {
            "sub_bass": 0.8,
            "bass": 0.9,
            "low_mid": 1.0,
            "mid": 1.1,
            "high_mid": 1.2,
            "high": 1.3,
        }

        # Adjust based on consciousness level
        consciousness_factor = params.consciousness_level / 13.0
        base_mod = base_factors[band_name]

        return base_mod * (1 + consciousness_factor * params.modulation_depth)
