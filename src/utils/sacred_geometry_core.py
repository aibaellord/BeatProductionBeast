"""
SacredGeometryCore - Advanced Sacred Geometry Mathematics for Audio Production

This module provides a comprehensive set of tools for implementing sacred geometry
principles in audio production, focusing on the golden ratio (phi), Fibonacci sequences,
harmonic resonances, and fractal patterns in frequency space.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union, Callable
import logging

# Constants
PHI = (1 + 5 ** 0.5) / 2  # Golden ratio ≈ 1.618033988749895
SCHUMANN_RESONANCE = 7.83  # Hz - Earth's fundamental resonant frequency
SOLFEGGIO_FREQUENCIES = {
    "UT": 396,   # Liberating guilt and fear
    "RE": 417,   # Undoing situations and facilitating change
    "MI": 528,   # Transformation and miracles (DNA repair)
    "FA": 639,   # Connecting/relationships
    "SOL": 741,  # Awakening intuition
    "LA": 852,   # Returning to spiritual order
}
CONSCIOUSNESS_LEVELS = {
    "delta": (0.5, 4),      # Deep sleep
    "theta": (4, 8),        # Drowsiness/meditation
    "alpha": (8, 14),       # Relaxed/reflective
    "beta": (14, 30),       # Alert/active
    "gamma": (30, 100),     # Higher consciousness
}


class SacredGeometryCore:
    """
    Core class implementing sacred geometry principles for audio production.
    
    This class provides methods for generating Fibonacci sequences, calculating
    phi-based harmonic resonances, applying golden ratio relationships to frequencies,
    and creating fractal patterns with self-similarity across scales.
    """
    
    def __init__(self, base_frequency: float = SCHUMANN_RESONANCE, 
                 sample_rate: int = 44100, 
                 consciousness_level: str = "alpha"):
        """
        Initialize the SacredGeometryCore with base parameters.
        
        Args:
            base_frequency: The fundamental frequency to base calculations on (default: Schumann resonance)
            sample_rate: Audio sample rate in Hz
            consciousness_level: Target consciousness level for frequency transformations
        """
        self.base_frequency = base_frequency
        self.sample_rate = sample_rate
        self.consciousness_level = consciousness_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize cached values
        self._fibonacci_cache = {0: 0, 1: 1}  # Memoization for Fibonacci
        self.logger.info(f"SacredGeometryCore initialized with base frequency {base_frequency}Hz")
        
    def fibonacci(self, n: int) -> int:
        """
        Calculate the nth Fibonacci number efficiently using memoization.
        
        Args:
            n: Position in the Fibonacci sequence (0-based)
            
        Returns:
            The nth Fibonacci number
            
        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError("Fibonacci is not defined for negative indices")
            
        if n in self._fibonacci_cache:
            return self._fibonacci_cache[n]
            
        # Calculate using dynamic programming approach
        self._fibonacci_cache[n] = self.fibonacci(n-1) + self.fibonacci(n-2)
        return self._fibonacci_cache[n]
    
    def fibonacci_sequence(self, length: int) -> List[int]:
        """
        Generate a Fibonacci sequence of specified length.
        
        Args:
            length: Number of Fibonacci numbers to generate
            
        Returns:
            List of Fibonacci numbers
        """
        return [self.fibonacci(i) for i in range(length)]
    
    def golden_ratio_powers(self, n: int) -> List[float]:
        """
        Generate a sequence of phi raised to different powers.
        
        Args:
            n: Number of powers to generate
            
        Returns:
            List of phi^i for i in range(n)
        """
        return [PHI ** i for i in range(n)]
    
    def phi_based_frequency(self, base: float, steps: int) -> float:
        """
        Calculate a frequency that is related to the base by the golden ratio.
        
        Args:
            base: Base frequency in Hz
            steps: Number of phi steps (positive or negative)
            
        Returns:
            New frequency harmonically related via phi
        """
        return base * (PHI ** steps)
    
    def generate_phi_harmonic_series(self, base_freq: float, 
                                     num_harmonics: int = 12) -> List[float]:
        """
        Generate a harmonic series based on phi relationships.
        
        Args:
            base_freq: Base frequency in Hz
            num_harmonics: Number of harmonics to generate
            
        Returns:
            List of frequencies forming a phi-based harmonic series
        """
        harmonics = [base_freq]
        
        # Generate positive phi harmonics
        for i in range(1, num_harmonics // 2 + 1):
            harmonics.append(self.phi_based_frequency(base_freq, i))
        
        # Generate negative phi harmonics (subharmonics)
        for i in range(1, num_harmonics // 2 + 1):
            harmonics.append(self.phi_based_frequency(base_freq, -i))
            
        return sorted(harmonics)
    
    def golden_ratio_frequency_division(self, frequency_range: Tuple[float, float]) -> Tuple[float, float, float]:
        """
        Divide a frequency range according to the golden ratio.
        
        Args:
            frequency_range: Tuple of (min_frequency, max_frequency)
            
        Returns:
            Tuple of (min_freq, golden_point, max_freq)
        """
        min_freq, max_freq = frequency_range
        span = max_freq - min_freq
        
        # Calculate the golden point
        golden_point = max_freq - (span / PHI)
        
        return min_freq, golden_point, max_freq
    
    def schumann_resonance_harmonics(self, num_harmonics: int = 7) -> List[float]:
        """
        Generate the Schumann resonance harmonics.
        
        The Schumann resonances are a set of spectrum peaks in the extremely low frequency 
        portion of the Earth's electromagnetic field spectrum.
        
        Args:
            num_harmonics: Number of harmonics to generate
            
        Returns:
            List of Schumann resonance harmonic frequencies
        """
        # Schumann resonances follow approximate formula: f_n ≈ 7.83 * sqrt(n*(n+1)/2)
        return [SCHUMANN_RESONANCE * math.sqrt(n * (n + 1) / 2) for n in range(1, num_harmonics + 1)]
    
    def apply_quantum_coherence(self, frequency: float, 
                                coherence_factor: float = 0.5) -> float:
        """
        Apply quantum coherence factor to a frequency.
        
        Args:
            frequency: Base frequency to adjust
            coherence_factor: Factor between 0 and 1 representing coherence strength
            
        Returns:
            Adjusted frequency with quantum coherence applied
        """
        if not 0 <= coherence_factor <= 1:
            raise ValueError("Coherence factor must be between 0 and 1")
            
        # Calculate quantum resonant frequency using Schumann as carrier
        resonant_freq = frequency * (1 + coherence_factor * (SCHUMANN_RESONANCE / frequency) ** 0.5)
        
        # Apply phi-based harmonic adjustment
        phi_adjustment = coherence_factor * (PHI - 1)
        
        return resonant_freq * (1 + phi_adjustment)
    
    def generate_phi_rhythm_pattern(self, 
                                   length: int, 
                                   base_pulse: float = 1.0) -> List[float]:
        """
        Generate a rhythm pattern based on phi relationships.
        
        Args:
            length: Length of the pattern
            base_pulse: Base pulse length in seconds
            
        Returns:
            List of pulse lengths forming a phi-based rhythm
        """
        fib_seq = self.fibonacci_sequence(length + 2)[2:]  # Skip first two (0, 1)
        
        # Create pattern of pulses based on Fibonacci ratios
        pattern = []
        for i in range(length):
            # Use ratio of consecutive Fibonacci numbers which approaches phi
            if i > 0:
                ratio = fib_seq[i] / fib_seq[i-1]
            else:
                ratio = PHI
                
            pattern.append(base_pulse * ratio)
            
        return pattern
    
    def create_fractal_pattern(self, 
                              base_pattern: List[float], 
                              depth: int, 
                              scale_factor: float = PHI) -> List[float]:
        """
        Create a fractal pattern with self-similarity across scales.
        
        Args:
            base_pattern: The seed pattern to build the fractal from
            depth: Number of recursive iterations
            scale_factor: Scaling factor between iterations (default: phi)
            
        Returns:
            List representing the fractal pattern
        """
        if depth <= 0:
            return base_pattern
            
        pattern = base_pattern.copy()
        
        for _ in range(depth):
            new_pattern = []
            for val in pattern:
                # For each element, add a scaled version of the entire pattern
                scaled = [val * item / scale_factor for item in base_pattern]
                new_pattern.extend(scaled)
            pattern = new_pattern
            
        return pattern
    
    def calculate_consciousness_frequency(self, 
                                        target_level: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate a consciousness-enhancing frequency range.
        
        Args:
            target_level: Target consciousness level (delta, theta, alpha, beta, gamma)
            
        Returns:
            Tuple of (center_frequency, bandwidth)
        """
        if target_level is None:
            target_level = self.consciousness_level
            
        if target_level not in CONSCIOUSNESS_LEVELS:
            raise ValueError(f"Unknown consciousness level: {target_level}")
            
        lower, upper = CONSCIOUSNESS_LEVELS[target_level]
        center = (lower + upper) / 2
        bandwidth = upper - lower
        
        # Apply golden ratio adjustment to center frequency
        phi_center = center * PHI
        
        return phi_center, bandwidth
    
    def generate_sacred_frequency_matrix(self, 
                                        base_freq: float,
                                        matrix_size: int = 3) -> np.ndarray:
        """
        Generate a matrix of sacred frequencies based on phi relationships.
        
        Args:
            base_freq: Base frequency in Hz
            matrix_size: Size of the matrix (n x n)
            
        Returns:
            numpy array of frequencies with golden ratio relationships
        """
        matrix = np.zeros((matrix_size, matrix_size))
        
        for i in range(matrix_size):
            for j in range(matrix_size):
                # Create a 2D field of frequencies with phi relationships
                # The distance from center determines the phi power
                x_dist = i - (matrix_size // 2)
                y_dist = j - (matrix_size // 2)
                distance = math.sqrt(x_dist**2 + y_dist**2)
                
                # Use phi^distance as the scalar
                matrix[i, j] = base_freq * (PHI ** distance)
                
        return matrix
    
    def create_golden_arrangement(self, 
                                 frequencies: List[float], 
                                 duration: float) -> Dict[float, float]:
        """
        Create an arrangement of frequencies over time following golden ratio proportions.
        
        Args:
            frequencies: List of frequencies to arrange
            duration: Total duration in seconds
            
        Returns:
            Dictionary mapping start times to frequencies
        """
        arrangement = {}
        
        # Get golden ratio points for the duration
        points = [0]
        remaining = duration
        current = 0
        
        while remaining > 0.01:  # Threshold for minimum segment
            golden_segment = remaining / PHI
            current += golden_segment
            points.append(current)
            remaining -= golden_segment
            
        # Map frequencies to points
        for i, freq in enumerate(frequencies):
            if i < len(points) - 1:
                arrangement[points[i]] = freq
                
        return arrangement
    
    def phi_based_tuning(self, 
                        reference_freq: float = 432.0) -> List[float]:
        """
        Create a phi-based musical tuning system.
        
        Args:
            reference_freq: Reference frequency (A4) in Hz
            
        Returns:
            List of 12 frequencies for an octave in phi-based tuning
        """
        # Use powers of phi to determine frequency ratios
        # Normalize to fit within standard 12-tone octave
        ratios = [(PHI ** (i/12)) % 2 for i in range(12)]
        
        # Sort to get ascending scale
        ratios.sort()
        
        # Apply to reference frequency
        return [reference_freq * ratio for ratio in ratios]
    
    def apply_schumann_modulation(self, 
                                signal: np.ndarray, 
                                modulation_depth: float = 0.1) -> np.ndarray:
        """
        Apply a modulation based on Schumann resonance.
        
        Args:
            signal: Input audio signal
            modulation_depth: Depth of modulation (0-1)
            
        Returns:
            Modulated signal
        """
        # Generate modulation signal using Schumann resonance
        t = np.arange(len(signal)) / self.sample_rate
        modulation = np.sin(2 * np.pi * SCHUMANN_RESONANCE * t)
        
        # Apply modulation
        modulated = signal * (1 + modulation_depth * modulation)
        
        return modulated
    
    def calculate_consciousness_alignment(self, 
                                        frequencies: List[float]) -> float:
        """
        Calculate how well a set of frequencies aligns with consciousness-enhancing ranges.
        
        Args:
            frequencies: List of frequencies to evaluate
            
        Returns:
            Alignment score between 0-1 (higher is better aligned)
        """
        # Get the target consciousness range
        center, bandwidth = self.calculate_consciousness_frequency()
        lower = center - bandwidth/2
        upper = center + bandwidth/2
        
        # Count frequencies in the target range
        in_range_count = sum(1 for f in frequencies if lower <= f <= upper)
        
        # Calculate phi-alignment for frequencies
        phi_alignment = sum(1 for f in frequencies 
                          for n in range(-3, 4)  # Check nearby phi relationships
                          if 0.98 <= f / (self.base_frequency * (PHI ** n)) <= 1.02)
        
        # Calculate combined score
        score = (0.5 * in_range_count / len(frequencies) + 
                0.5 * phi_alignment / len(frequencies))
        
        return min(score, 1.0)  # Cap at 1.0

