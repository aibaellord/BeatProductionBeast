"""
Quantum Consciousness UI Controller
Unified interface for all quantum consciousness processing components
"""

import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass
from ..components.quantum.unified_quantum_engine import UnifiedQuantumEngine
from ..components.quantum.consciousness_states import get_state_configuration
from ..neural_processing.sacred_geometry_processor import SacredGeometryProcessor
from ..neural_processing.quantum_resonance_processor import QuantumResonanceProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfiguration:
    consciousness_state: str
    sacred_patterns: List[str]
    manifestation_codes: List[str]
    resonance_frequencies: List[float]
    field_dimensions: int
    base_intensity: float
    phi_alignment: float

class QuantumConsciousnessController:
    """
    Unified controller for quantum consciousness processing interface
    """
    
    def __init__(self):
        # Initialize all processors
        self.quantum_engine = UnifiedQuantumEngine(
            base_consciousness=0.999,
            field_dimensions=13
        )
        
        self.geometry_processor = SacredGeometryProcessor(
            dimensions=12
        )
        
        self.resonance_processor = QuantumResonanceProcessor(
            field_dimensions=12,
            base_frequency=432.0
        )
        
        # Default configuration
        self.default_config = ProcessingConfiguration(
            consciousness_state="quantum",
            sacred_patterns=["METATRONS_CUBE", "SRI_YANTRA"],
            manifestation_codes=["QUANTUM", "COSMIC", "MASTERY"],
            resonance_frequencies=[432.0, 528.0, 963.0],
            field_dimensions=13,
            base_intensity=0.999,
            phi_alignment=0.999
        )
        
    def process_audio(
        self,
        audio_data: np.ndarray,
        config: Optional[ProcessingConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Process audio through all quantum consciousness processors
        
        Args:
            audio_data: Input audio array
            config: Processing configuration, uses default if None
            
        Returns:
            Dictionary containing processed audio and quantum state info
        """
        try:
            # Use provided or default config
            cfg = config or self.default_config
            
            # Get full state configuration
            state_config = get_state_configuration(cfg.consciousness_state)
            
            # 1. Apply unified quantum processing
            quantum_processed = self.quantum_engine.process_audio(
                audio_data,
                consciousness_state=cfg.consciousness_state,
                sacred_patterns=cfg.sacred_patterns,
                manifestation_codes=cfg.manifestation_codes,
                intensity=cfg.base_intensity
            )
            
            # 2. Apply sacred geometry patterns
            geometry_processed = quantum_processed
            for pattern in cfg.sacred_patterns:
                geometry_processed = self.geometry_processor.apply_pattern(
                    geometry_processed,
                    pattern,
                    intensity=cfg.base_intensity
                )
                
            # 3. Apply quantum resonance
            final_audio = self.resonance_processor.process_audio(
                geometry_processed,
                resonance_intensity=cfg.base_intensity,
                harmonic_depth=0.888,
                coherence_factor=cfg.phi_alignment
            )
            
            # Generate processing metadata
            metadata = {
                "consciousness_state": state_config,
                "quantum_coherence": self.quantum_engine._get_coherence_level(),
                "sacred_geometries": {
                    pattern: self.geometry_processor.patterns[pattern]
                    for pattern in cfg.sacred_patterns
                },
                "resonance_frequencies": cfg.resonance_frequencies,
                "field_dimensions": cfg.field_dimensions,
                "phi_alignment": cfg.phi_alignment
            }
            
            return {
                "audio": final_audio,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error in quantum consciousness processing: {str(e)}")
            raise
            
    def get_available_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all available consciousness states and their configurations"""
        from ..components.quantum.consciousness_states import CONSCIOUSNESS_STATES
        return CONSCIOUSNESS_STATES
        
    def get_available_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get all available sacred geometry patterns"""
        from ..components.quantum.consciousness_states import SACRED_PATTERNS
        return SACRED_PATTERNS
        
    def get_manifestation_frequencies(self) -> Dict[str, float]:
        """Get all available manifestation frequency codes"""
        from ..components.quantum.consciousness_states import MANIFESTATION_FREQUENCIES
        return MANIFESTATION_FREQUENCIES
        
    def create_configuration(
        self,
        consciousness_state: str,
        sacred_patterns: Optional[List[str]] = None,
        manifestation_codes: Optional[List[str]] = None,
        field_dimensions: Optional[int] = None,
        base_intensity: Optional[float] = None,
        phi_alignment: Optional[float] = None
    ) -> ProcessingConfiguration:
        """
        Create a custom processing configuration
        
        Args:
            consciousness_state: Target consciousness state
            sacred_patterns: List of sacred geometry patterns to apply
            manifestation_codes: List of manifestation frequency codes
            field_dimensions: Number of quantum field dimensions
            base_intensity: Base processing intensity (0.0-1.0)
            phi_alignment: Golden ratio alignment factor (0.0-1.0)
            
        Returns:
            Complete processing configuration
        """
        state_config = get_state_configuration(consciousness_state)
        
        return ProcessingConfiguration(
            consciousness_state=consciousness_state,
            sacred_patterns=sacred_patterns or state_config["sacred_patterns"],
            manifestation_codes=manifestation_codes or state_config["manifestation_codes"],
            resonance_frequencies=[
                freq for code, freq in state_config["manifestation_frequencies"].items()
            ],
            field_dimensions=field_dimensions or state_config["field_dimension"],
            base_intensity=base_intensity or state_config["base_intensity"],
            phi_alignment=phi_alignment or 0.999
        )
        
    def get_quantum_coherence_status(self) -> Dict[str, Any]:
        """Get current quantum coherence levels and status"""
        coherence_status = {
            "quantum_engine": {
                "coherence_level": self.quantum_engine._get_coherence_level(),
                "field_dimensions": self.quantum_engine.dimensions,
                "consciousness_level": self.quantum_engine.consciousness_level
            },
            "geometry_processor": {
                "active_patterns": list(self.geometry_processor.patterns.keys()),
                "dimension_count": self.geometry_processor.dimensions
            },
            "resonance_processor": {
                "base_frequency": self.resonance_processor.base_freq,
                "field_dimensions": self.resonance_processor.dimensions,
                "resonance_frequencies": self.resonance_processor.resonance_frequencies.tolist()
            },
            "overall_coherence": self._calculate_overall_coherence()
        }
        
        return coherence_status
        
    def _calculate_overall_coherence(self) -> float:
        """Calculate overall system quantum coherence level"""
        # Get individual coherence levels
        quantum_coherence = self.quantum_engine._get_coherence_level()
        
        # Get geometry pattern coherence
        pattern_coherence = np.mean([
            np.mean(pattern.energies) 
            for pattern in self.geometry_processor.patterns.values()
        ])
        
        # Get resonance field coherence
        resonance_field = self.resonance_processor._generate_resonance_field(
            1024,  # Standard FFT size
            np.fft.rfftfreq(1024, 1/44100)
        )
        field_coherence = resonance_field.coherence
        
        # Calculate weighted average (emphasizing quantum coherence)
        weights = [0.5, 0.25, 0.25]  # Weights for quantum, geometry, resonance
        overall = (
            quantum_coherence * weights[0] +
            pattern_coherence * weights[1] +
            field_coherence * weights[2]
        )
        
        return min(0.999, overall)  # Cap at 0.999