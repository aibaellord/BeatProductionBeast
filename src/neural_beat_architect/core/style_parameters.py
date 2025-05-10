from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class StyleParameters:
    tempo_range: Optional[Tuple[float, float]] = None
    consciousness_depth: Optional[float] = None
    quantum_coherence_factor: Optional[float] = None
    golden_ratio_alignment: Optional[float] = None
    fractal_dimension: Optional[float] = None
    rhythm_syncopation: Optional[float] = None
    polyrhythm_factor: Optional[float] = None
    groove_intensity: Optional[float] = None
    schumann_resonance_align: Optional[bool] = None
    phi_harmonic_structure: Optional[bool] = None
    solfeggio_integration: Optional[bool] = None
    emotional_attunement: Optional[Dict[str, float]] = field(default_factory=dict)
    mental_state_targeting: Optional[Dict[str, float]] = field(default_factory=dict)
