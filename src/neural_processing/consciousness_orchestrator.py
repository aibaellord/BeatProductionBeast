"""
Neural Consciousness Orchestrator
Coordinates quantum field processing with sacred geometry patterns
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from .sacred_geometry_processor import SacredGeometryProcessor
from .consciousness_field_integrator import ConsciousnessFieldIntegrator
from .consciousness_manifestation import ConsciousnessManifestationProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    sacred_patterns: List[str]
    field_intensity: float
    consciousness_depth: int
    quantum_iterations: int
    harmonic_threshold: float

class ConsciousnessOrchestrator:
    """
    Orchestrates consciousness processing across multiple dimensions
    """
    
    def __init__(
        self,
        dimensions: int = 13,
        base_frequency: float = 432.0,
        sampling_rate: int = 44100
    ):
        self.dimensions = dimensions
        self.base_freq = base_frequency
        self.sampling_rate = sampling_rate
        
        # Initialize processors
        self.sacred_processor = SacredGeometryProcessor(
            dimensions=dimensions,
            base_frequency=base_frequency
        )
        
        self.field_integrator = ConsciousnessFieldIntegrator(
            dimensions=dimensions,
            sampling_rate=sampling_rate
        )
        
        self.manifestation = ConsciousnessManifestationProcessor(
            dimensions=dimensions
        )
        
        # Processing defaults
        self.default_config = ProcessingConfig(
            sacred_patterns=["METATRONS_CUBE", "FLOWER_OF_LIFE"],
            field_intensity=0.85,
            consciousness_depth=7,
            quantum_iterations=3,
            harmonic_threshold=0.02
        )
        
    def process_consciousness_field(
        self,
        audio_data: np.ndarray,
        config: Optional[ProcessingConfig] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process audio through consciousness field
        
        Args:
            audio_data: Input audio array
            config: Processing configuration
            
        Returns:
            Tuple of (processed_audio, processing_metrics)
        """
        if config is None:
            config = self.default_config
            
        metrics = {}
        
        try:
            # Apply sacred geometry patterns
            sacred_audio, sacred_metrics = self.sacred_processor.apply_sacred_patterns(
                audio_data,
                config.sacred_patterns,
                config.field_intensity
            )
            metrics["sacred_geometry"] = sacred_metrics
            
            # Integrate consciousness field
            field_audio, field_metrics = self.field_integrator.integrate_field(
                sacred_audio,
                depth=config.consciousness_depth,
                iterations=config.quantum_iterations
            )
            metrics["field_integration"] = field_metrics
            
            # Apply consciousness manifestation
            manifested_audio, manifestation_metrics = self.manifestation.process_manifestation(
                field_audio,
                harmonic_threshold=config.harmonic_threshold
            )
            metrics["manifestation"] = manifestation_metrics
            
            # Apply final harmonic balancing
            result = self._balance_harmonics(manifested_audio)
            
            # Calculate overall consciousness coherence
            metrics["coherence"] = self._calculate_coherence(
                result,
                sacred_metrics,
                field_metrics,
                manifestation_metrics
            )
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Error in consciousness processing: {str(e)}")
            raise
            
    def _balance_harmonics(self, audio: np.ndarray) -> np.ndarray:
        """Balance harmonic content of processed audio"""
        spectrum = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1.0/self.sampling_rate)
        
        # Apply harmonic balancing
        for i, freq in enumerate(freqs):
            if i < len(spectrum):
                # Find fundamental frequency region
                if 20 <= freq <= 120:
                    # Enhance fundamentals slightly
                    spectrum[i] *= 1.15
                    
                # Balance overtone series
                elif 120 < freq <= 3500:
                    harmonic_factor = 1.0 / (1 + (freq - 120) / 3380)
                    spectrum[i] *= (1 + harmonic_factor * 0.3)
                    
                # Gentle rolloff of higher frequencies
                else:
                    rolloff = np.exp(-(freq - 3500) / 10000)
                    spectrum[i] *= rolloff
                    
        result = np.fft.irfft(spectrum)
        
        # Normalize
        max_amp = np.max(np.abs(result))
        if max_amp > 0.98:
            result = result * (0.98 / max_amp)
            
        return result
        
    def _calculate_coherence(
        self,
        audio: np.ndarray,
        sacred_metrics: Dict,
        field_metrics: Dict,
        manifestation_metrics: Dict
    ) -> float:
        """Calculate overall consciousness coherence"""
        # Average sacred geometry effectiveness
        sacred_coherence = np.mean(list(sacred_metrics.values()))
        
        # Field integration stability
        field_coherence = field_metrics.get("stability", 0.0)
        
        # Manifestation clarity
        manifestation_coherence = manifestation_metrics.get("clarity", 0.0)
        
        # Calculate spectral coherence
        spectrum = np.fft.rfft(audio)
        spectral_coherence = np.mean(np.abs(spectrum)) / np.max(np.abs(spectrum))
        
        # Weighted combination
        total_coherence = (
            sacred_coherence * 0.3 +
            field_coherence * 0.3 +
            manifestation_coherence * 0.2 +
            spectral_coherence * 0.2
        )
        
        return float(total_coherence)