"""
Neural Processing Module

This module provides neural network processing capabilities including
model loading, prediction, and feature extraction for audio data.
"""

from .feature_extractor import FeatureExtractor
# Import main classes for easier access
from .model_loader import ModelLoader
from .neural_model import NeuralModel
from .predictor import Predictor
from .quantum_field_processor import (ConsciousnessAmplifier,
                                      MultidimensionalFieldProcessor)
from .sacred_enhancer import QuantumSacredEnhancer

# Define what's available for import with "from neural_processing import *"
__all__ = [
    "ModelLoader",
    "Predictor",
    "FeatureExtractor",
    "MultidimensionalFieldProcessor",
    "ConsciousnessAmplifier",
    "QuantumSacredEnhancer",
]
