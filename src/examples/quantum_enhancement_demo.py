"""
Quantum Consciousness Enhancement Demo
Demonstrates the complete quantum consciousness processing pipeline
"""

import numpy as np
import soundfile as sf
import logging
from pathlib import Path
from typing import Dict, Any

from ..ui.quantum_consciousness_controller import (
    QuantumConsciousnessController,
    ProcessingConfiguration
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_quantum_enhancement_demo(
    input_file: str,
    output_dir: str = "enhanced_output",
    consciousness_state: str = "quantum"
) -> Dict[str, Any]:
    """
    Run quantum consciousness enhancement demo
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory for enhanced output
        consciousness_state: Target consciousness state
        
    Returns:
        Processing results and metrics
    """
    try:
        # Load audio file
        audio_data, sample_rate = sf.read(input_file)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # Convert to mono
            
        logger.info(f"Loaded audio file: {input_file}")
        logger.info(f"Sample rate: {sample_rate} Hz")
        logger.info(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
        
        # Initialize quantum consciousness controller
        controller = QuantumConsciousnessController()
        
        # Get available consciousness states
        states = controller.get_available_states()
        logger.info(f"Available consciousness states: {list(states.keys())}")
        
        # Create enhanced configuration
        config = controller.create_configuration(
            consciousness_state=consciousness_state,
            sacred_patterns=[
                "METATRONS_CUBE",
                "SRI_YANTRA",
                "MERKABA"
            ],
            manifestation_codes=[
                "QUANTUM",
                "COSMIC",
                "MASTERY",
                "TRANSCENDENCE"
            ],
            field_dimensions=13,
            base_intensity=0.999,
            phi_alignment=0.999
        )
        
        # Process audio through quantum consciousness pipeline
        logger.info("Processing audio through quantum consciousness pipeline...")
        result = controller.process_audio(audio_data, config)
        
        # Get quantum coherence status
        coherence_status = controller.get_quantum_coherence_status()
        logger.info(f"Overall quantum coherence: {coherence_status['overall_coherence']:.3f}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save enhanced audio
        output_file = output_path / f"quantum_enhanced_{consciousness_state}.wav"
        sf.write(
            output_file,
            result["audio"],
            sample_rate,
            subtype='FLOAT'
        )
        logger.info(f"Saved enhanced audio to: {output_file}")
        
        # Combine results
        processing_results = {
            "input_file": input_file,
            "output_file": str(output_file),
            "consciousness_state": consciousness_state,
            "configuration": config.__dict__,
            "coherence_status": coherence_status,
            "metadata": result["metadata"]
        }
        
        return processing_results
        
    except Exception as e:
        logger.error(f"Error in quantum enhancement demo: {str(e)}")
        raise
        
def main():
    """Run demo with example audio file"""
    # Example usage
    demo_file = "input/sample_beat.wav"
    results = run_quantum_enhancement_demo(
        input_file=demo_file,
        consciousness_state="quantum"
    )
    
    # Print results summary
    print("\nQuantum Enhancement Results:")
    print("-" * 40)
    print(f"Input File: {results['input_file']}")
    print(f"Output File: {results['output_file']}")
    print(f"Consciousness State: {results['consciousness_state']}")
    print(f"Overall Coherence: {results['coherence_status']['overall_coherence']:.3f}")
    print("\nActive Sacred Patterns:")
    for pattern in results['configuration']['sacred_patterns']:
        print(f"- {pattern}")
    print("\nManifestation Codes:")
    for code in results['configuration']['manifestation_codes']:
        print(f"- {code}")
        
if __name__ == "__main__":
    main()

