"""
Neural Processing Module

This module provides neural network processing capabilities including
model loading, prediction, and feature extraction for audio data.
"""

# Import main classes for easier access
from .model_loader import ModelLoader
from .predictor import Predictor
from .feature_extractor import FeatureExtractor
from .quantum_field_processor import MultidimensionalFieldProcessor, ConsciousnessAmplifier
from .sacred_enhancer import QuantumSacredEnhancer
from .neural_model import NeuralModel

# Define what's available for import with "from neural_processing import *"
__all__ = ['ModelLoader', 'Predictor', 'FeatureExtractor', 
           'MultidimensionalFieldProcessor', 'ConsciousnessAmplifier',
           'QuantumSacredEnhancer']

