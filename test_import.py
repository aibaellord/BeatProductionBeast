#!/usr/bin/env python3
"""
Test script to verify the BeatProductionBeast package structure.
Attempts to import various modules and classes from the package and reports success/failure.
"""

import importlib
import sys
from typing import List, Tuple


def test_import(module_path: str, class_names: List[str] = None) -> Tuple[bool, str]:
    """
    Test importing a module and optionally specific classes from that module.
    
    Args:
        module_path: The full module path to import
        class_names: Optional list of class names to import from the module
        
    Returns:
        Tuple of (success, message)
    """
    try:
        module = importlib.import_module(module_path)
        if not class_names:
            return True, f"✅ Successfully imported {module_path}"
        
        for class_name in class_names:
            try:
                getattr(module, class_name)
            except AttributeError:
                return False, f"❌ Failed to import {class_name} from {module_path}"
        
        classes_str = ", ".join(class_names)
        return True, f"✅ Successfully imported {classes_str} from {module_path}"
    
    except ImportError as e:
        return False, f"❌ Failed to import {module_path}: {e}"


def run_import_tests():
    """Run all import tests and print results."""
    # List of modules and classes to test
    # Format: (module_path, [class_names] or None for module-only test)
    imports_to_test = [
        # Main package
        ("BeatProductionBeast", None),
        
        # Modules
        ("BeatProductionBeast.neural_beat_architect", None),
        ("BeatProductionBeast.audio_engine", ["AudioProcessor", "SoundGenerator", "MixerInterface"]),
        ("BeatProductionBeast.neural_processing", ["ModelLoader", "Predictor", "FeatureExtractor"]),
        ("BeatProductionBeast.beat_generation", None),
        ("BeatProductionBeast.fusion_generator", None),
        ("BeatProductionBeast.harmonic_enhancement", None),
        ("BeatProductionBeast.pattern_recognition", None),
        ("BeatProductionBeast.style_analysis", None),
        ("BeatProductionBeast.utils", None),
        
        # Direct imports from the main package (these should be exposed in src/__init__.py)
        ("BeatProductionBeast", ["AudioProcessor", "ModelLoader", "FeatureExtractor"]),
    ]
    
    print("\n===== TESTING BEATPRODUCTIONBEAST PACKAGE STRUCTURE =====\n")
    
    all_successful = True
    for module_path, class_names in imports_to_test:
        success, message = test_import(module_path, class_names)
        print(message)
        if not success:
            all_successful = False
    
    print("\n===== IMPORT TEST SUMMARY =====")
    if all_successful:
        print("✅ All imports successful! Package structure is working correctly.")
    else:
        print("❌ Some imports failed. Please check the package structure.")
        print("\nPossible issues:")
        print("  - Missing __init__.py files in some directories")
        print("  - PYTHONPATH not set correctly")
        print("  - Package not installed or not installed in development mode")
        print("  - Class names don't match expected names or aren't properly exported")
    
    print("\nTo fix PYTHONPATH issues, make sure to:")
    print("  1. Activate the virtual environment")
    print("  2. Install the package in development mode: pip install -e .")
    print("  3. Ensure your activate script correctly sets PYTHONPATH")
    

if __name__ == "__main__":
    run_import_tests()

