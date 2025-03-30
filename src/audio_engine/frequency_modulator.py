"""
Frequency Modulation Module

This module provides frequency modulation capabilities for consciousness enhancement,
including binaural beats generation and brainwave entrainment with advanced sacred geometry principles.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from enum import Enum
import math

from .core import AudioProcessor
from src.utils.sacred_geometry_core import SacredGeometryCore

class BrainwaveRange(Enum):
    """Brainwave frequency ranges associated with different mental states."""
    DELTA = (0.5, 4)    # Deep sleep, healing
    THETA = (4, 8)      # Meditation, creativity
    ALPHA = (8, 14)     # Relaxation, calmness
    BETA = (14, 30)     # Focus, alertness
    GAMMA = (30, 100)   # Higher cognition, insight
    LAMBDA = (100, 200) # Advanced states of consciousness
    EPSILON = (200, 400) # Transcendental states


class SolfeggioFrequency(Enum):
    """Ancient Solfeggio frequencies with spiritual significance."""
    UT = 396   # Liberating guilt and fear
    RE = 417   # Undoing situations and facilitating change
    MI = 528   # Transformation and miracles (DNA repair)
    FA = 639   # Connecting/relationships
    SOL = 741  # Awakening intuition
    LA = 852   # Returning to spiritual order
    SI = 963   # Awakening and returning to oneness


class SacredGeometryRatio(Enum):
    """Sacred geometry ratios for frequency relationships."""
    PHI = 1.618033988749895  # Golden ratio
    PI = math.pi             # Circle's circumference to diameter
    SQRT2 = math.sqrt(2)     # Diagonal of a square
    SQRT3 = math.sqrt(3)     # Altitude of an equilateral triangle
    SQRT5 = math.sqrt(5)     # Diagonal of a golden rectangle
    FIBONACCI_RATIO = 0.618033988749895  # 1/φ
    SCHUMANN = 7.83          # Earth's resonant frequency


class FrequencyModulator:
    """
    Advanced frequency modulation tool for consciousness-enhancing audio.
    
    This class enables the creation of complex frequency patterns including
    binaural beats, brainwave entrainment, and integration of sacred geometry
    and Solfeggio frequencies.
    """
    
    def __init__(self, audio_processor: AudioProcessor):
        """
        Initialize the FrequencyModulator.
        
        Args:
            audio_processor: The core audio processor to use for waveform manipulation
        """
        self.audio_processor = audio_processor
        self.sample_rate = audio_processor.sample_rate
        
        # Initialize frequency maps
        self._init_frequency_maps()
    
    def _init_frequency_maps(self) -> None:
        """Initialize internal frequency maps and relationships."""
        # Create mappings for sacred geometry frequencies
        self.sacred_geometry_map = {
            'phi': 1.618033988749895,
            'inverse_phi': 0.618033988749895,
            'pi': math.pi,
            'e': math.e,
            'sqrt2': math.sqrt(2),
            'sqrt3': math.sqrt(3),
            'sqrt5': math.sqrt(5)
        }
        
        # Map for chakra frequencies (Hz)
        self.chakra_frequencies = {
            'root': 256,       # C
            'sacral': 288,     # D
            'solar_plexus': 320, # E
            'heart': 341.3,    # F
            'throat': 384,     # G
            'third_eye': 426.7, # A
            'crown': 480       # B
        }
        
        # Brainwave entrainment carrier frequencies
        self.entrainment_carriers = {
            'relaxation': 432,  # Relaxation base frequency
            'focus': 528,       # Focus/clarity base frequency
            'meditation': 256,  # Meditation base frequency
            'sleep': 174,       # Sleep inducing base frequency
            'healing': 285,     # Physical healing base frequency
            'balance': 396,     # Balancing base frequency
            'awareness': 963    # Spiritual awareness base frequency
        }
    
    def generate_binaural_beat(
        self, 
        base_frequency: float, 
        beat_frequency: float, 
        duration: float,
        volume: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a binaural beat audio array.
        
        Creates two separate frequency tracks where the difference between
        the frequencies matches the requested beat frequency.
        
        Args:
            base_frequency: The carrier frequency (Hz)
            beat_frequency: The desired beat frequency (Hz) - difference between left and right
            duration: Length of the audio in seconds
            volume: Amplitude of the generated audio (0.0-1.0)
            
        Returns:
            Tuple containing (left_channel, right_channel) as numpy arrays
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        # Left channel uses the base frequency
        left_channel = volume * np.sin(2 * np.pi * base_frequency * t)
        
        # Right channel uses base frequency + beat frequency
        right_channel = volume * np.sin(2 * np.pi * (base_frequency + beat_frequency) * t)
        
        return left_channel, right_channel
    
    def generate_brainwave_entrainment(
        self,
        target_state: Union[str, BrainwaveRange],
        duration: float,
        modulation_depth: float = 0.3,
        carrier_frequency: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate audio designed to entrain the brain to a specific brainwave state.
        
        Args:
            target_state: The desired brainwave state (can be a BrainwaveRange enum or string name)
            duration: Length of the audio in seconds
            modulation_depth: Depth of the frequency modulation (0.0-1.0)
            carrier_frequency: Optional base frequency for carrier wave
            
        Returns:
            Stereo audio array (left and right channels)
        """
        # Determine target frequency range
        if isinstance(target_state, str):
            target_state = BrainwaveRange[target_state.upper()]
        
        # Get the middle of the range as the target frequency
        target_freq = sum(target_state.value) / 2
        
        # Choose carrier frequency if not specified
        if carrier_frequency is None:
            if target_state == BrainwaveRange.DELTA:
                carrier_frequency = self.entrainment_carriers['sleep']
            elif target_state == BrainwaveRange.THETA:
                carrier_frequency = self.entrainment_carriers['meditation']
            elif target_state == BrainwaveRange.ALPHA:
                carrier_frequency = self.entrainment_carriers['relaxation']
            elif target_state == BrainwaveRange.BETA:
                carrier_frequency = self.entrainment_carriers['focus']
            elif target_state == BrainwaveRange.GAMMA:
                carrier_frequency = self.entrainment_carriers['awareness']
            else:
                carrier_frequency = 432.0  # Default carrier
        
        # Create binaural beat at the target frequency
        left_channel, right_channel = self.generate_binaural_beat(
            carrier_frequency, target_freq, duration, 0.5
        )
        
        # Apply sacred geometry modulation
        left_channel = self._apply_sacred_geometry_modulation(
            left_channel, 
            carrier_frequency, 
            'phi', 
            modulation_depth,
            duration
        )
        
        right_channel = self._apply_sacred_geometry_modulation(
            right_channel, 
            carrier_frequency + target_freq, 
            'inverse_phi', 
            modulation_depth,
            duration
        )
        
        # Combine into stereo
        return np.vstack((left_channel, right_channel)).T
    
    def _apply_sacred_geometry_modulation(
        self,
        audio: np.ndarray,
        base_freq: float,
        ratio_key: str,
        depth: float,
        duration: float
    ) -> np.ndarray:
        """
        Apply sacred geometry-based frequency modulation to an audio array.
        
        Args:
            audio: Input audio array
            base_freq: Base frequency for modulation
            ratio_key: Key in sacred_geometry_map for the ratio to use
            depth: Modulation depth (0.0-1.0)
            duration: Duration in seconds
            
        Returns:
            Modulated audio array
        """
        if ratio_key not in self.sacred_geometry_map:
            raise ValueError(f"Unknown sacred geometry ratio: {ratio_key}")
        
        ratio = self.sacred_geometry_map[ratio_key]
        modulation_freq = base_freq / ratio
        
        # Create modulation signal
        t = np.linspace(0, duration, len(audio), False)
        modulation = depth * np.sin(2 * np.pi * modulation_freq * t)
        
        # Apply modulation
        return audio * (1.0 + modulation)
    
    def generate_solfeggio_frequency(
        self,
        tone: Union[str, SolfeggioFrequency],
        duration: float,
        volume: float = 0.5
    ) -> np.ndarray:
        """
        Generate a pure tone at one of the Solfeggio frequencies.
        
        Args:
            tone: The Solfeggio tone to generate (can be enum or string name)
            duration: Length of the audio in seconds
            volume: Amplitude of the generated audio (0.0-1.0)
            
        Returns:
            Audio array containing the Solfeggio frequency
        """
        # Get the frequency
        if isinstance(tone, str):
            tone = SolfeggioFrequency[tone.upper()]
        
        frequency = tone.value
        
        # Generate the tone
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        audio = volume * np.sin(2 * np.pi * frequency * t)
        
        return audio
    
    def create_fibonacci_harmony(
        self,
        base_frequency: float,
        duration: float,
        num_harmonics: int = 8,
        volume: float = 0.5
    ) -> np.ndarray:
        """
        Create a harmonic series based on the Fibonacci sequence.
        
        Args:
            base_frequency: The fundamental frequency
            duration: Length of the audio in seconds
            num_harmonics: Number of Fibonacci harmonics to include
            volume: Overall volume of the resulting audio
            
        Returns:
            Audio array with Fibonacci harmonics
        """
        # Generate Fibonacci sequence
        fib_sequence = [1, 1]
        for i in range(2, num_harmonics + 2):
            fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
        
        # Create audio array
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        # Start with silence
        audio = np.zeros(num_samples)
        
        # Add each harmonic with decreasing volume
        for i, fib in enumerate(fib_sequence[:num_harmonics]):
            harmonic_freq = base_frequency * (fib / fib_sequence[0])
            harmonic_volume = volume * (1.0 / (i + 1))
            audio += harmonic_volume * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 1.0:
            audio = audio / max_val * volume
            
        return audio
    
    def create_sacred_geometry_progression(
        self,
        base_frequency: float,
        duration: float,
        progression_type: str = 'phi',
        steps: int = 7,
        volume: float = 0.5
    ) -> np.ndarray:
        """
        Create a frequency progression based on sacred geometry ratios.
        
        Args:
            base_frequency: Starting frequency
            duration: Total duration of the progression
            progression_type: Type of sacred geometry to use ('phi', 'pi', 'sqrt2', etc.)
            steps: Number of steps in the progression
            volume: Overall volume level
            
        Returns:
            Audio array containing the geometric progression
        """
        if progression_type not in self.sacred_geometry_map:
            raise ValueError(f"Unknown sacred geometry type: {progression_type}")
        
        ratio = self.sacred_geometry_map[progression_type]
        step_duration = duration / steps
        
        # Create empty output array
        num_samples = int(duration * self.sample_rate)
        audio = np.zeros(num_samples)
        
        # Generate each step
        current_freq = base_frequency
        for i in range(steps):
            # Calculate start and end sample for this step
            start_sample = int(i * step_duration * self.sample_rate)
            end_sample = int((i + 1) * step_duration * self.sample_rate)
            if end_sample > num_samples:
                end_sample = num_samples
                
            # Generate tone for this step
            step_samples = end_sample - start_sample
            t = np.linspace(0, step_duration, step_samples, False)
            step_audio = volume * np.sin(2 * np.pi * current_freq * t)
            
            # Apply fade in/out to avoid clicks
            fade_samples = min(int(0.01 * self.sample_rate), step_samples // 4)
            if fade_samples > 0:
                # Fade in
                step_audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # Fade out
                step_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Add to main audio array
            audio[start_sample:end_sample] = step_audio
            
            # Update frequency for next step
            current_freq *= ratio
        
        return audio
    
    def create_chakra_alignment_sequence(
        self,
        duration_per_chakra: float,
        include_transition: bool = True,
        transition_duration: float = 3.0
    ) -> np.ndarray:
        """
        Create an audio sequence that cycles through the 7 chakra frequencies.
        
        Args:
            duration_per_chakra: Duration to spend on each chakra frequency
            include_transition: Whether to include smooth transitions between frequencies
            transition_duration: Duration of transition between chakras if enabled
            
        Returns:
            Audio array containing the chakra alignment sequence
        """
        chakras = ['root', 'sacral', 'solar_plexus', 'heart', 'throat', 'third_eye', 'crown']
        
        if include_transition:
            # With transitions, each chakra plays for (duration_per_chakra - transition_duration)
            full_sequence = np.array([])
            for i, chakra in enumerate(chakras):
                frequency = self._get_chakra_frequency(chakra)
                chakra_audio = self.generate_pure_tone(frequency, duration_per_chakra - transition_duration)
                
                if i < len(chakras) - 1:
                    # Generate transition to next chakra
                    next_frequency = self._get_chakra_frequency(chakras[i+1])
                    transition = self.generate_frequency_transition(
                        frequency, next_frequency, transition_duration
                    )
                    full_sequence = np.concatenate([full_sequence, chakra_audio, transition])
                else:
                    # Last chakra has no transition after it
                    full_sequence = np.concatenate([full_sequence, chakra_audio])
            
            return full_sequence
        else:
            # Without transitions, simply concatenate each chakra tone
            full_sequence = np.array([])
            for chakra in chakras:
                frequency = self._get_chakra_frequency(chakra)
                chakra_audio = self.generate_pure_tone(frequency, duration_per_chakra)
                full_sequence = np.concatenate([full_sequence, chakra_audio])
            
            return full_sequence
