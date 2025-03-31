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
                # Check if this is not the last chakra
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

    def enhance_harmonics(
        self,
        audio: np.ndarray,
        strength: float = 0.5,
        harmonic_profile: str = 'golden',
        preserve_timbre: bool = True
    ) -> np.ndarray:
        """
        Enhance the harmonic content of audio with phi-optimized processing.
        
        This method analyzes the spectral content of the audio and enhances
        harmonically related frequencies based on sacred geometry principles.
        
        Args:
            audio: Input audio array (can be mono or stereo)
            strength: Enhancement strength (0.0-1.0)
            harmonic_profile: Type of harmonic profile to apply ('golden', 'fibonacci', 'chakra')
            preserve_timbre: Whether to preserve the original timbre while enhancing harmonics
            
        Returns:
            Audio with enhanced harmonic content
        """
        # Clamp strength parameter
        strength = max(0.0, min(1.0, strength))
        
        # Check if input is stereo (2D) or mono (1D)
        is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2
        
        if is_stereo:
            # Process each channel separately
            left_channel = audio[:, 0]
            right_channel = audio[:, 1]
            
            left_enhanced = self._enhance_channel_harmonics(
                left_channel, strength, harmonic_profile, preserve_timbre
            )
            right_enhanced = self._enhance_channel_harmonics(
                right_channel, strength, harmonic_profile, preserve_timbre
            )
            
            # Recombine channels
            return np.column_stack((left_enhanced, right_enhanced))
        else:
            # Process mono audio
            return self._enhance_channel_harmonics(
                audio, strength, harmonic_profile, preserve_timbre
            )
    
    def _enhance_channel_harmonics(
        self, 
        channel: np.ndarray, 
        strength: float,
        harmonic_profile: str,
        preserve_timbre: bool
    ) -> np.ndarray:
        """
        Enhance harmonics for a single audio channel.
        
        Args:
            channel: Single channel audio data
            strength: Enhancement strength
            harmonic_profile: Type of harmonic profile
            preserve_timbre: Whether to preserve original timbre
            
        Returns:
            Enhanced single channel audio
        """
        # Calculate FFT
        n_fft = min(8192, len(channel))
        hop_length = n_fft // 4
        
        # Use simple overlapping windows for processing
        enhanced_audio = np.zeros_like(channel)
        window = np.hanning(n_fft)
        
        # Process each frame
        for i in range(0, len(channel) - n_fft, hop_length):
            # Extract frame and apply window
            frame = channel[i:i+n_fft] * window
            
            # Calculate spectrum
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            # Apply harmonic enhancement
            enhanced_magnitude = self._apply_harmonic_enhancement(
                magnitude, strength, harmonic_profile
            )
            
            # If preserving timbre, use a blend of original and enhanced magnitudes
            if preserve_timbre:
                blend_factor = 0.3 + (0.7 * strength)  # Adjust based on strength
                enhanced_magnitude = (
                    (1 - blend_factor) * magnitude + 
                    blend_factor * enhanced_magnitude
                )
            
            # Reconstruct spectrum and convert back to time domain
            enhanced_spectrum = enhanced_magnitude * np.exp(1j * phase)
            enhanced_frame = np.fft.irfft(enhanced_spectrum)
            
            # Overlap-add to output
            enhanced_audio[i:i+n_fft] += enhanced_frame
        
        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(enhanced_audio))
        if max_amplitude > 0.99:
            enhanced_audio = enhanced_audio * (0.99 / max_amplitude)
        
        return enhanced_audio
    
    def _apply_harmonic_enhancement(
        self,
        magnitude: np.ndarray,
        strength: float,
        profile: str
    ) -> np.ndarray:
        """
        Apply harmonic enhancement to magnitude spectrum.
        
        Args:
            magnitude: Magnitude spectrum
            strength: Enhancement strength
            profile: Harmonic profile type
            
        Returns:
            Enhanced magnitude spectrum
        """
        enhanced = np.copy(magnitude)
        n_bins = len(magnitude)
        
        # Find strongest frequency components
        peaks = []
        for i in range(1, n_bins - 1):
            if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                if magnitude[i] > 0.1 * np.max(magnitude):  # Threshold
                    peaks.append(i)
        
        # Limit the number of peaks we consider
        if len(peaks) > 20:
            # Sort by magnitude and take the strongest ones
            peaks.sort(key=lambda x: magnitude[x], reverse=True)
            peaks = peaks[:20]  # Take top 20
        
        # Apply different harmonic profiles
        if profile == 'golden':
            ratio = self.sacred_geometry_map['phi']
            boost_factor = 2.0 * strength
        elif profile == 'fibonacci':
            fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21]
            ratio = 1.0  # Will use fibonacci sequence directly
            boost_factor = 2.5 * strength
        elif profile == 'chakra':
            # Use chakra frequency relationships
            ratio = 1.125  # Musical fifth relationship
            boost_factor = 1.8 * strength
        else:
            # Default to phi-based
            ratio = self.sacred_geometry_map['phi']
            boost_factor = 2.0 * strength
        
        # Enhance harmonics for each detected peak
        for peak in peaks:
            if peak == 0:
                continue  # Skip DC component
                
            if profile == 'fibonacci':
                # Enhance at fibonacci multipliers
                for fib in fib_sequence[2:]:  # Skip first two (1, 1)
                    harmonic_bin = int(peak * fib)
                    if 0 < harmonic_bin < n_bins:
                        # Boost the harmonic with tapering strength
                        boost = boost_factor * (1.0 / fib) * magnitude[peak]
                        # Apply smooth window around the harmonic
                        window_size = 3
                        for w in range(-window_size, window_size + 1):
                            if 0 <= harmonic_bin + w < n_bins:
                                window_weight = 1.0 - (abs(w) / (window_size + 1))
                                enhanced[harmonic_bin + w] += boost * window_weight
            else:
                # For golden ratio and other profiles
                # Calculate harmonic series based on the profile
                for h in range(2, 8):  # Harmonics 2 through 7
                    # Calculate frequency bin for this harmonic
                    if profile == 'chakra':
                        harmonic_bin = int(peak * h)  # Integer harmonics for chakra
                    else:
                        harmonic_bin = int(peak * ratio * h)
                        
                    if 0 < harmonic_bin < n_bins:
                        # Calculate boost amount (decreasing with harmonic number)
                        boost = boost_factor * (1.0 / h) * magnitude[peak]
                        
                        # Apply smooth window around the harmonic
                        window_size = 3
                        for w in range(-window_size, window_size + 1):
                            if 0 <= harmonic_bin + w < n_bins:
                                window_weight = 1.0 - (abs(w) / (window_size + 1))
                                enhanced[harmonic_bin + w] += boost * window_weight
        
        # Apply sacred geometry enhancement curve
        phi = self.sacred_geometry_map['phi']
        inv_phi = self.sacred_geometry_map['inverse_phi']
        
        # Apply overall spectral shaping based on sacred geometry
        for i in range(n_bins):
            # Create phi-based curve that enhances certain frequency regions
            position = i / n_bins
            sacred_curve = 1.0 + (0.2 * strength * np.sin(2 * np.pi * phi * position)**2)
            
            # Apply additional enhancement at golden ratio points in spectrum
            for j in range(1, 4):  # Several phi-related points
                phi_point = inv_phi * j
                if abs(position - phi_point) < 0.05:  # Within 5% of phi points
                    sacred_curve += 0.15 * strength * (1.0 - abs(position - phi_point) / 0.05)
            
            enhanced[i] *= sacred_curve
        
        return enhanced
        
    def _get_chakra_frequency(self, chakra: str) -> float:
        """
        Get the frequency for a specific chakra.
        
        Args:
            chakra: Name of the chakra (root, sacral, solar_plexus, heart, throat, third_eye, crown)
            
        Returns:
            Frequency in Hz for the requested chakra
        """
        if chakra not in self.chakra_frequencies:
            raise ValueError(f"Unknown chakra: {chakra}")
            
        return self.chakra_frequencies[chakra]
        
    def generate_frequency_transition(
        self,
        start_frequency: float,
        end_frequency: float,
        duration: float,
        volume: float = 0.5,
        transition_curve: str = 'exponential'
    ) -> np.ndarray:
        """
        Generate a smooth transition between two frequencies.
        
        Args:
            start_frequency: Starting frequency in Hz
            end_frequency: Ending frequency in Hz
            duration: Duration of the transition in seconds
            volume: Amplitude of the generated audio (0.0-1.0)
            transition_curve: Type of transition curve ('linear', 'exponential', 'logarithmic', 'sinusoidal')
            
        Returns:
            Audio array containing the frequency transition
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        # Create the transition curve
        if transition_curve == 'linear':
            # Linear transition
            freq_values = np.linspace(start_frequency, end_frequency, num_samples)
        elif transition_curve == 'logarithmic':
            # Logarithmic transition (faster at the beginning)
            log_space = np.logspace(0, 1, num_samples) - 1
            freq_values = start_frequency + (end_frequency - start_frequency) * (log_space / 9)
        elif transition_curve == 'sinusoidal':
            # Sinusoidal transition (smooth S-curve)
            phase = np.linspace(0, np.pi, num_samples)
            s_curve = (1 - np.cos(phase)) / 2
            freq_values = start_frequency + (end_frequency - start_frequency) * s_curve
        else:  # Default to exponential
            # Exponential transition (slower at beginning, faster at end)
            exp_space = np.exp(np.linspace(0, 1, num_samples)) - 1
            freq_values = start_frequency + (end_frequency - start_frequency) * (exp_space / (np.e - 1))
        
        # Use frequency values to compute instantaneous phase
        phase = np.cumsum(2 * np.pi * freq_values / self.sample_rate)
        
        # Generate the waveform using the computed phase
        audio = volume * np.sin(phase)
        
        # Apply fade in/out to avoid clicks
        fade_samples = min(int(0.01 * self.sample_rate), num_samples // 10)
        if fade_samples > 0:
            # Fade in
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return audio
    
    def generate_pure_tone(
        self,
        frequency: float,
        duration: float,
        volume: float = 0.5,
        waveform: str = 'sine'
    ) -> np.ndarray:
        """
        Generate a pure tone at the specified frequency.
        
        Args:
            frequency: Frequency of the tone in Hz
            duration: Duration of the tone in seconds
            volume: Amplitude of the generated audio (0.0-1.0)
            waveform: Type of waveform ('sine', 'square', 'triangle', 'sawtooth')
            
        Returns:
            Audio array containing the pure tone
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate the waveform
        if waveform == 'square':
            audio = volume * np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform == 'triangle':
            audio = volume * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
        elif waveform == 'sawtooth':
            audio = volume * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        else:  # Default to sine
            audio = volume * np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out to avoid clicks
        fade_samples = min(int(0.01 * self.sample_rate), num_samples // 10)
        if fade_samples > 0:
            # Fade in
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return audio
