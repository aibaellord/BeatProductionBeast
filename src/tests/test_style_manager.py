#!/usr/bin/env python3
"""
Test file for StyleManagerUI class that demonstrates key functionality:
1. Creating a StyleManagerUI instance
2. Loading a style
3. Applying a quick transform
4. Using phi-optimization
5. Changing consciousness level
"""

import os
import sys
from pathlib import Path

# Add src directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.neural_beat_architect.core.architect import StyleParameters
from src.neural_beat_architect.core.style_factory import StyleFactory
from src.ui.style_manager_ui import StyleManagerUI


def main():
    """
    Main function demonstrating StyleManagerUI usage with simple examples.
    """
    print("===== StyleManagerUI Testing =====")

    # 1. Creating a StyleManagerUI instance
    print("\n1. Creating StyleManagerUI instance...")
    # Initialize with default parameters
    style_manager = StyleManagerUI()
    print(f"StyleManagerUI created: {style_manager}")

    # 2. Loading a style
    print("\n2. Loading a style...")
    # Load a lo-fi style
    style_name = "lo-fi"
    style = style_manager.load_style(style_name)
    print(f"Loaded style: {style_name}")
    print(
        f"Style parameters: BPM={style.bpm}, Swing={style.swing:.2f}, Intensity={style.intensity:.2f}"
    )

    # 3. Applying a quick transform
    print("\n3. Applying a quick transform...")
    # Transform to a more energetic version
    transformed_style = style_manager.apply_quick_transform(style, "energize")
    print(f"Applied 'energize' transform")
    print(f"Original BPM: {style.bpm}, New BPM: {transformed_style.bpm}")
    print(
        f"Original Intensity: {style.intensity:.2f}, New Intensity: {transformed_style.intensity:.2f}"
    )

    # 4. Using phi-optimization
    print("\n4. Using phi-optimization...")
    # Optimize harmony and rhythm parameters using golden ratio
    optimized_style = style_manager.apply_phi_optimization(
        transformed_style, parameters=["harmony", "rhythm"]
    )
    print(f"Applied phi-optimization to harmony and rhythm")
    print(f"Original harmony: {transformed_style.harmony_complexity:.2f}")
    print(f"Phi-optimized harmony: {optimized_style.harmony_complexity:.2f}")

    # 5. Changing consciousness level
    print("\n5. Changing consciousness level...")
    # Increase consciousness level (scale: 1-13, with Fibonacci numbers preferred)
    elevated_style = style_manager.set_consciousness_level(optimized_style, 8)
    print(f"Changed consciousness level to 8")
    print(f"Original consciousness level: {optimized_style.consciousness_level}")
    print(f"New consciousness level: {elevated_style.consciousness_level}")
    print(f"Notice changes in sacred parameters:")
    print(f"  - Sacred frequency ratio: {elevated_style.sacred_frequency_ratio:.3f}")
    print(f"  - Mental state targeting: {elevated_style.mental_state_targeting}")
    print(f"  - Neural coherence: {elevated_style.neural_coherence:.2f}")

    # Summary
    print("\n===== Final Style Overview =====")
    print(f"Style: {style_name} (transformed, phi-optimized, consciousness level 8)")
    print(f"BPM: {elevated_style.bpm}")
    print(f"Key: {elevated_style.key}")
    print(f"Swing: {elevated_style.swing:.2f}")
    print(f"Intensity: {elevated_style.intensity:.2f}")
    print(f"Consciousness Level: {elevated_style.consciousness_level}")
    print(f"Sacred Frequency Ratio: {elevated_style.sacred_frequency_ratio:.3f}")

    print("\n===== Test Complete =====")


if __name__ == "__main__":
    main()
