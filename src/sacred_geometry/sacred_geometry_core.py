"""
Sacred Geometry Core Module

This module provides a comprehensive implementation of sacred geometry principles
applied to audio processing. It offers tools to transform, analyze, and generate
audio content based on mathematical patterns found in nature and consciousness studies.

The SacredGeometryCore class provides methods for:
- Golden ratio (phi) based frequency calculations
- Fibonacci sequence applications in audio processing
- Sacred geometry pattern implementation for rhythm and harmony
- Frequency relationships based on sacred ratios
- Consciousness-aligned audio transformation

These tools enable the creation of harmonically rich audio content that aligns with
natural mathematical proportions found throughout the universe.
"""

import math
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class GeometryPattern(Enum):
    """Enum for different sacred geometry patterns"""

    FLOWER_OF_LIFE = 0
    METATRONS_CUBE = 1
    SRI_YANTRA = 2
    TORUS = 3
    VESICA_PISCES = 4
    SEED_OF_LIFE = 5
    FRUIT_OF_LIFE = 6
    PLATONIC_SOLIDS = 7


class SacredGeometryCore:
    """
    A comprehensive toolkit for applying sacred geometry principles to audio processing.

    This class provides methods for transforming audio based on golden ratio (phi),
    Fibonacci sequences, and other sacred geometry patterns. It enables the creation
    of harmonically rich audio content aligned with natural mathematical proportions.
    """

    def __init__(self):
        """Initialize the SacredGeometryCore with fundamental constants"""
        # The golden ratio (phi)
        self.phi = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895

        # Fundamental frequency for A4 (440Hz is standard)
        self.base_frequency = 432.0  # Consciousness-aligned base frequency

        # First 20 numbers in the Fibonacci sequence
        self.fibonacci_sequence = self._generate_fibonacci_sequence(20)

        # Sacred ratios from various traditions
        self.sacred_ratios = {
            "perfect_fifth": 3 / 2,
            "perfect_fourth": 4 / 3,
            "major_third": 5 / 4,
            "minor_third": 6 / 5,
            "octave": 2 / 1,
            "golden_ratio": self.phi,
            "sqrt2": math.sqrt(2),
            "sqrt3": math.sqrt(3),
            "pi_phi": math.pi / self.phi,
        }

        # Sacred intervals in cents (100 cents = 1 semitone)
        self.sacred_intervals = {
            "phi_interval": 1200 * math.log2(self.phi),
            "perfect_fifth": 702,
            "perfect_fourth": 498,
            "major_third": 386,
            "minor_third": 316,
            "pythagorean_comma": 23.46,
            "syntonic_comma": 21.51,
        }

    def _generate_fibonacci_sequence(self, n: int) -> List[int]:
        """
        Generate the first n numbers in the Fibonacci sequence.

        Args:
            n: Number of Fibonacci numbers to generate

        Returns:
            List of the first n Fibonacci numbers
        """
        if n <= 0:
            return []
        if n == 1:
            return [0]
        if n == 2:
            return [0, 1]

        sequence = [0, 1]
        for i in range(2, n):
            sequence.append(sequence[i - 1] + sequence[i - 2])

        return sequence

    def generate_phi_based_frequency(
        self, base_freq: float, iterations: int
    ) -> List[float]:
        """
        Generate a series of frequencies based on the golden ratio.

        Args:
            base_freq: Starting frequency in Hz
            iterations: Number of frequencies to generate

        Returns:
            List of phi-related frequencies
        """
        frequencies = [base_freq]

        for i in range(1, iterations):
            # Ascending phi relationship
            if i % 2 == 1:
                next_freq = frequencies[-1] * self.phi
            # Descending phi relationship
            else:
                next_freq = frequencies[-1] / self.phi

            frequencies.append(next_freq)

        return frequencies

    def create_fibonacci_rhythm(
        self, beats_per_minute: float, sequence_length: int
    ) -> List[float]:
        """
        Create rhythm patterns based on Fibonacci sequence time intervals.

        Args:
            beats_per_minute: BPM of the rhythm
            sequence_length: Number of beats to generate

        Returns:
            List of beat timings in seconds
        """
        # Calculate the duration of one beat in seconds
        beat_duration = 60.0 / beats_per_minute

        # Generate a fibonacci sequence for the rhythm
        fib_sequence = self._generate_fibonacci_sequence(sequence_length + 2)[2:]

        # Convert fibonacci numbers to rhythm intervals
        rhythm_pattern = []
        current_time = 0.0

        for i in range(sequence_length):
            current_time += (fib_sequence[i] % 8 + 1) * beat_duration / 4
            rhythm_pattern.append(current_time)

        return rhythm_pattern

    def apply_sacred_ratio(self, frequency: float, ratio_name: str) -> float:
        """
        Apply a sacred ratio to transform a frequency.

        Args:
            frequency: Input frequency in Hz
            ratio_name: Name of the sacred ratio to apply

        Returns:
            Transformed frequency

        Raises:
            ValueError: If ratio_name is not recognized
        """
        if ratio_name not in self.sacred_ratios:
            raise ValueError(f"Unknown sacred ratio: {ratio_name}")

        return frequency * self.sacred_ratios[ratio_name]

    def generate_harmonic_series(
        self, fundamental: float, num_harmonics: int
    ) -> List[float]:
        """
        Generate harmonic series from a fundamental frequency.

        Args:
            fundamental: Fundamental frequency in Hz
            num_harmonics: Number of harmonics to generate

        Returns:
            List of harmonic frequencies
        """
        return [fundamental * (n + 1) for n in range(num_harmonics)]

    def phi_weighted_frequencies(
        self, freq_list: List[float], center_weight: float = 1.0
    ) -> np.ndarray:
        """
        Apply phi-based weighting to a list of frequencies.

        Args:
            freq_list: List of input frequencies
            center_weight: Weight of the center frequency

        Returns:
            Array of weighted frequencies
        """
        weights = np.zeros(len(freq_list))
        center_idx = len(freq_list) // 2

        for i in range(len(freq_list)):
            # Calculate distance from center
            distance = abs(i - center_idx)

            # Apply phi-based weighting
            if distance == 0:
                weights[i] = center_weight
            else:
                weights[i] = center_weight / (self.phi**distance)

        # Normalize weights
        weights = weights / np.sum(weights)

        return np.array(freq_list) * weights

    def generate_sacred_scale(
        self, root_frequency: float, pattern: str = "phi"
    ) -> List[float]:
        """
        Generate a scale based on sacred geometry principles.

        Args:
            root_frequency: Root note frequency in Hz
            pattern: Pattern type to use ('phi', 'fibonacci', 'pythagorean')

        Returns:
            List of frequencies in the sacred scale

        Raises:
            ValueError: If pattern is not recognized
        """
        if pattern == "phi":
            # Golden ratio based scale
            scale = [root_frequency]
            for i in range(1, 8):
                next_note = scale[-1] * (self.phi ** (1 / 7))
                scale.append(next_note)

        elif pattern == "fibonacci":
            # Fibonacci relationship based scale
            ratios = [1, 2 / 1, 3 / 2, 5 / 3, 8 / 5, 13 / 8, 21 / 13, 34 / 21]
            scale = [root_frequency * ratio for ratio in ratios]

        elif pattern == "pythagorean":
            # Pythagorean tuning
            ratios = [1, 9 / 8, 81 / 64, 4 / 3, 3 / 2, 27 / 16, 243 / 128, 2]
            scale = [root_frequency * ratio for ratio in ratios]

        else:
            raise ValueError(f"Unknown scale pattern: {pattern}")

        return scale

    def generate_phi_rhythm_matrix(
        self, measures: int, beats_per_measure: int
    ) -> np.ndarray:
        """
        Generate a rhythm matrix based on phi relationships.

        Args:
            measures: Number of measures
            beats_per_measure: Number of beats per measure

        Returns:
            2D numpy array with rhythm intensities (0-1)
        """
        total_beats = measures * beats_per_measure
        matrix = np.zeros((measures, beats_per_measure))

        for measure in range(measures):
            for beat in range(beats_per_measure):
                # Position in the sequence
                position = measure * beats_per_measure + beat

                # Phi-based intensity calculation
                phi_position = position * self.phi
                intensity = 0.5 + 0.5 * math.sin(phi_position)

                # Apply Fibonacci modulation
                fib_mod = self.fibonacci_sequence[
                    position % len(self.fibonacci_sequence)
                ]
                intensity *= 0.5 + 0.5 * (fib_mod % 5) / 5

                matrix[measure, beat] = intensity

        return matrix

    def apply_geometry_pattern(
        self, audio_array: np.ndarray, pattern: GeometryPattern
    ) -> np.ndarray:
        """
        Apply a sacred geometry pattern transformation to audio data.

        Args:
            audio_array: Input audio data
            pattern: Sacred geometry pattern to apply

        Returns:
            Transformed audio array
        """
        # Get the appropriate transformation function based on the pattern
        transform_func = self._get_geometry_transform(pattern)

        # Apply the transformation
        return transform_func(audio_array)

    def _get_geometry_transform(
        self, pattern: GeometryPattern
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get the transformation function for a specific geometry pattern.

        Args:
            pattern: Sacred geometry pattern

        Returns:
            Transformation function
        """
        if pattern == GeometryPattern.FLOWER_OF_LIFE:
            return self._apply_flower_of_life
        elif pattern == GeometryPattern.METATRONS_CUBE:
            return self._apply_metatrons_cube
        elif pattern == GeometryPattern.SRI_YANTRA:
            return self._apply_sri_yantra
        elif pattern == GeometryPattern.TORUS:
            return self._apply_torus
        elif pattern == GeometryPattern.VESICA_PISCES:
            return self._apply_vesica_pisces
        elif pattern == GeometryPattern.SEED_OF_LIFE:
            return self._apply_seed_of_life
        elif pattern == GeometryPattern.FRUIT_OF_LIFE:
            return self._apply_fruit_of_life
        elif pattern == GeometryPattern.PLATONIC_SOLIDS:
            return self._apply_platonic_solids
        else:
            # Default to identity transformation
            return lambda x: x

    def _apply_flower_of_life(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Flower of Life pattern transformation to audio.

        The Flower of Life pattern creates overlapping circle patterns,
        translated to audio as overlapping harmonic enhancements.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        result = np.copy(audio)

        # Apply 6-fold symmetry transformation (hexagonal pattern)
        for i in range(1, 7):
            phase = 2 * math.pi * i / 6
            # Create a phase-shifted copy
            shifted = np.roll(audio, int(len(audio) / 6 * i))
            # Apply harmonic weighting based on position in pattern
            weight = 0.5 + 0.5 * math.sin(phase)
            result += shifted * weight * 0.2

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_metatrons_cube(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Metatron's Cube pattern to audio.

        This pattern incorporates all Platonic solids, translated to audio
        as a complex harmonic structure based on whole-number ratios.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        result = np.copy(audio)
        fft_data = np.fft.rfft(audio)

        # Apply transformations based on platonic solid frequencies
        platonic_ratios = [1, 4 / 3, 3 / 2, 5 / 3, 2]

        for ratio in platonic_ratios:
            # Shift and scale the frequency domain
            shifted_fft = np.zeros_like(fft_data)
            shift_amount = int(len(fft_data) * ratio / 3)

            # Apply frequency domain transformation
            for i in range(len(fft_data) - shift_amount):
                shifted_fft[i + shift_amount] = fft_data[i] * (1 / ratio) * 0.15

            # Transform back to time domain and add to result
            shifted_audio = np.fft.irfft(shifted_fft, len(audio))
            result += shifted_audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_sri_yantra(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Sri Yantra sacred geometry pattern to audio.

        The Sri Yantra pattern uses interlocking triangles, translated to audio as
        a series of harmonic resonances based on triangular number relationships.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        # Apply triangular number series modulation
        triangle_nums = [n * (n + 1) // 2 for n in range(1, 10)]

        # Create copy of input audio
        result = np.copy(audio)

        # Split audio into segments
        num_segments = 9
        segment_length = len(audio) // num_segments

        # Apply triangular number based processing to each segment
        for i in range(num_segments):
            # Calculate start and end positions for this segment
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_segments - 1 else len(audio)

            # Get the triangle number for this segment
            tri_num = triangle_nums[i % len(triangle_nums)]

            # Apply modulation based on triangle number
            # The modulation increases in complexity with each triangle number
            phase_shift = (
                2 * math.pi * tri_num / 45
            )  # 45 is sum of first 9 triangle numbers

            # Generate modulation signal for this segment
            t = np.linspace(0, 2 * tri_num * math.pi, end - start)
            modulation = 0.5 * np.sin(t + phase_shift) + 0.5

            # Apply frequency-dependent modulation
            segment_fft = np.fft.rfft(audio[start:end])

            # Apply harmonic enhancement at phi-related frequencies
            freq_indices = [
                int(len(segment_fft) * (1 / self.phi) * n % len(segment_fft))
                for n in range(1, tri_num + 1)
            ]
            for idx in freq_indices:
                segment_fft[idx] *= 1 + 0.2 * tri_num / 9

            # Apply phase modulation based on Sri Yantra geometry
            phase = np.angle(segment_fft)
            phase_mod = phase + phase_shift * modulation[: len(phase)]

            # Reconstruct the frequency domain with modified phase
            magnitude = np.abs(segment_fft)
            segment_fft_mod = magnitude * np.exp(1j * phase_mod)

            # Transform back to time domain
            audio_mod = np.fft.irfft(segment_fft_mod, end - start)

            # Apply amplitude modulation
            result[start:end] = audio_mod * (1 + 0.15 * modulation[: len(audio_mod)])

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_torus(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Torus pattern transformation to audio.

        The Torus pattern creates a cyclical pattern with self-reference,
        translated to audio as cyclical frequency modulations and feedback.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        result = np.copy(audio)

        # Apply FFT to get frequency domain representation
        fft_data = np.fft.rfft(audio)

        # Create a torus-like frequency modulation
        # Torus has two characteristic cycles (major and minor)
        major_cycle = len(fft_data) // 3
        minor_cycle = int(major_cycle / self.phi)

        # Modulate frequency amplitudes based on torus geometry
        for i in range(len(fft_data)):
            # Create a modulation based on position in major and minor cycles
            major_phase = 2 * math.pi * (i % major_cycle) / major_cycle
            minor_phase = 2 * math.pi * (i % minor_cycle) / minor_cycle

            # Combine the phases in a torus-like pattern
            torus_mod = 0.5 + 0.25 * np.sin(major_phase) + 0.25 * np.sin(minor_phase)

            # Apply the modulation
            fft_data[i] *= torus_mod

        # Apply a feedback loop to simulate the self-referential nature of a torus
        # This creates a delayed copy of the signal that feeds back into itself
        feedback_ratio = 1 / self.phi
        feedback_delay = int(len(audio) * feedback_ratio) % len(audio)

        # Create delayed version
        delayed = np.roll(audio, feedback_delay)

        # Mix original and delayed with golden ratio proportions
        feedback_strength = 0.3
        result = result * (1 - feedback_strength) + delayed * feedback_strength

        # Transform modulated frequency data back to time domain
        freq_modulated = np.fft.irfft(fft_data, len(audio))

        # Combine frequency and time domain modulations
        result = 0.7 * result + 0.3 * freq_modulated

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_vesica_pisces(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Vesica Pisces pattern transformation to audio.

        The Vesica Pisces pattern creates overlapping circles where the center of each
        is on the circumference of the other, translated to audio as two overlapping
        frequency spectra with a specific sacred ratio relationship.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        result = np.copy(audio)

        # The Vesica Pisces is created by two overlapping circles
        # We'll simulate this by creating two overlapping versions of the audio

        # Calculate the phi-based offset for the second circle
        offset = int(len(audio) * (1 - 1 / self.phi)) % len(audio)

        # Create the second circle (offset version)
        circle2 = np.roll(audio, offset)

        # The "vesica" is the overlapping region
        # In audio terms, we'll create an effect that emphasizes frequencies common to both

        # Transform both signals to frequency domain
        fft_1 = np.fft.rfft(audio)
        fft_2 = np.fft.rfft(circle2)

        # Calculate magnitudes
        mag_1 = np.abs(fft_1)
        mag_2 = np.abs(fft_2)

        # The "vesica" is represented by the geometric mean of the two spectra
        vesica_mag = np.sqrt(mag_1 * mag_2)

        # Get the phases
        phase_1 = np.angle(fft_1)
        phase_2 = np.angle(fft_2)

        # Create a blended phase based on the root ratio
        vesica_phase = (phase_1 + phase_2) / 2

        # Create the new spectrum using the vesica magnitude and blended phase
        vesica_fft = vesica_mag * np.exp(1j * vesica_phase)

        # Transform back to time domain
        vesica_audio = np.fft.irfft(vesica_fft, len(audio))

        # Blend original and vesica-processed audio
        # The vesica in sacred geometry has specific proportions related to sqrt(3)
        vesica_ratio = (
            math.sqrt(3) / 2
        )  # Sacred proportion of vesica pisces height to width
        result = (1 - vesica_ratio) * audio + vesica_ratio * vesica_audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_seed_of_life(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Seed of Life pattern transformation to audio.

        The Seed of Life consists of seven circles arranged with six-fold symmetry,
        translated to audio as a seven-band harmonic enhancement with specific phase relationships.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        result = np.copy(audio)

        # The Seed of Life has 7 circles - we'll create 7 frequency bands
        num_bands = 7

        # Transform to frequency domain
        fft_data = np.fft.rfft(audio)
        fft_length = len(fft_data)

        # Create band boundaries based on phi relationships
        band_boundaries = [0]
        for i in range(1, num_bands):
            # Each band is related to the previous by the golden ratio
            boundary = int(
                band_boundaries[-1] + (fft_length - band_boundaries[-1]) / self.phi
            )
            band_boundaries.append(boundary)
        band_boundaries.append(fft_length)

        # Process each band according to the Seed of Life pattern
        for i in range(num_bands):
            start = band_boundaries[i]
            end = band_boundaries[i + 1]

            # Calculate phase shift based on position in the seed pattern
            # In the Seed of Life, circles are arranged with 60° (π/3) separations
            phase_shift = 2 * math.pi * i / num_bands

            # Apply phase shift to this frequency band
            band_data = fft_data[start:end]
            phases = np.angle(band_data)
            magnitudes = np.abs(band_data)

            # Enhance magnitudes at resonant frequencies
            # In the Seed of Life, certain intersections create resonant nodes
            if i > 0:  # Skip the base/center circle
                resonance_factor = 1 + 0.2 * math.sin(i * math.pi / 6)
                magnitudes *= resonance_factor

            # Apply the phase shift to create the sacred geometry pattern
            modified_phases = phases + phase_shift

            # Recombine magnitude and phase
            fft_data[start:end] = magnitudes * np.exp(1j * modified_phases)

        # Transform back to time domain
        processed_audio = np.fft.irfft(fft_data, len(audio))

        # Blend with original using golden ratio proportion
        phi_ratio = 1 / self.phi
        result = phi_ratio * audio + (1 - phi_ratio) * processed_audio

        # Normalize to prevent clipping
        max_val = np.max(np.abs(result))
        if max_val > 1.0:
            result = result / max_val

        return result

    def _apply_fruit_of_life(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Fruit of Life pattern transformation to audio.

        The Fruit of Life consists of 13 circles in a specific hexagonal arrangement,
        translated to audio as a 13-band frequency enhancement with harmonic relationships.

        Args:
            audio: Input audio array

        Returns:
            Transformed audio array
        """
        # The Fruit of Life has 13 circles arranged in a specific pattern
        num_circles = 13

        # Create an empty result array
        result = np.zeros_like(audio)

        # For each circle in the pattern, create a frequency-shifted version
        # The positions follow a hexagonal arrangement with specific proportions
        for i in range(num_circles):
            # Calculate the position in the pattern
            # The Fruit of Life has inner and outer rings of circles
            if i == 0:
                # Center circle
                shift_factor = 0
            elif i <= 6:
                # Inner ring - 6 circles
                angle = 2 * math.pi * (i - 1) / 6
                shift_factor = 0.3 * math.sin(angle)
            else:
                # Outer ring - 6 circles
                angle = 2 * math.pi * (i - 7) / 6 + (
                    math.pi / 6
                )  # Offset from inner ring
                shift_factor = 0.5 * math.sin
