"""
Neural Processing Module

This module provides neural network processing capabilities including
model loading, prediction, and feature extraction for audio data.
"""

# Import main classes for easier access
from .model_loader import ModelLoader
from .predictor import Predictor
from .feature_extractor import FeatureExtractor

# Define what's available for import with "from neural_processing import *"
__all__ = ['ModelLoader', 'Predictor', 'FeatureExtractor']

