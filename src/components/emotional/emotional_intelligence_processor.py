import numpy as np
import librosa
import tensorflow as tf
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, Any
import json
import os
from scipy.signal import butter, filtfilt
from scipy import stats

from src.preset.quantum_consciousness_engine import QuantumConsciousnessEngine, MultidimensionalAudioFrame
from src.utils.audio_processing import extract_instrumentals, apply_spectral_processing


class EmotionalDimension(Enum):
    """Emotional dimensions for analyzing and processing audio."""
    VALENCE = auto()  # Positive vs. negative emotion
    AROUSAL = auto()  # High energy vs. low energy
    DOMINANCE = auto()  # Feeling of control
    INTENSITY = auto()  # Emotional intensity
    COMPLEXITY = auto()  # Emotional complexity
    AUTHENTICITY = auto()  # Emotional authenticity
    VULNERABILITY = auto()  # Emotional vulnerability
    INTIMACY = auto()  # Emotional closeness
    TRANSCENDENCE = auto()  # Spiritual/transcendent quality


class EmotionalCategory(Enum):
    """Specific emotional categories for classification and generation."""
    JOY = auto()
    SADNESS = auto()
    ANGER = auto()
    FEAR = auto()
    SURPRISE = auto()
    DISGUST = auto()
    LOVE = auto()
    CONTENTMENT = auto()
    PRIDE = auto()
    SHAME = auto()
    GUILT = auto()
    ENVY = auto()
    JEALOUSY = auto()
    HOPE = auto()
    ANXIETY = auto()
    BOREDOM = auto()
    NOSTALGIA = auto()
    AWE = auto()
    EXCITEMENT = auto()
    CALM = auto()
    MELANCHOLY = auto()
    TRIUMPH = auto()
    LONGING = auto()
    GRATITUDE = auto()
    SERENITY = auto()
    ECSTASY = auto()
    CONTEMPLATIVE = auto()
    DETERMINED = auto()
    EMPOWERED = auto()
    VULNERABLE = auto()


@dataclass
class EmotionalSignature:
    """
    Represents the emotional characteristics of an audio sample.
    Contains both dimensional and categorical emotional data.
    """
    # Dimensional emotional values (0.0 to 1.0)
    dimensions: Dict[EmotionalDimension, float]
    
    # Categorical emotional probabilities (0.0 to 1.0)
    categories: Dict[EmotionalCategory, float]
    
    # Time-series emotional data
    temporal_dimensions: Optional[Dict[EmotionalDimension, np.ndarray]] = None
    temporal_categories: Optional[Dict[EmotionalCategory, np.ndarray]] = None
    
    # Metadata
    confidence: float = 0.0
    analysis_timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            "dimensions": {dim.name: value for dim, value in self.dimensions.items()},
            "categories": {cat.name: value for cat, value in self.categories.items()},
            "confidence": self.confidence,
            "analysis_timestamp": self.analysis_timestamp
        }
        
        # Handle temporal data if present
        if self.temporal_dimensions:
            result["temporal_dimensions"] = {
                dim.name: values.tolist() for dim, values in self.temporal_dimensions.items()
            }
            
        if self.temporal_categories:
            result["temporal_categories"] = {
                cat.name: values.tolist() for cat, values in self.temporal_categories.items()
            }
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EmotionalSignature':
        """Create an EmotionalSignature from a dictionary."""
        dimensions = {EmotionalDimension[key]: value for key, value in data["dimensions"].items()}
        categories = {EmotionalCategory[key]: value for key, value in data["categories"].items()}
        
        signature = cls(
            dimensions=dimensions,
            categories=categories,
            confidence=data.get("confidence", 0.0),
            analysis_timestamp=data.get("analysis_timestamp", 0.0)
        )
        
        # Load temporal data if present
        if "temporal_dimensions" in data:
            signature.temporal_dimensions = {
                EmotionalDimension[key]: np.array(values) 
                for key, values in data["temporal_dimensions"].items()
            }
            
        if "temporal_categories" in data:
            signature.temporal_categories = {
                EmotionalCategory[key]: np.array(values) 
                for key, values in data["temporal_categories"].items()
            }
            
        return signature
    
    def get_dominant_emotion(self) -> EmotionalCategory:
        """Returns the most prominent emotional category."""
        return max(self.categories.items(), key=lambda x: x[1])[0]
    
    def get_emotional_complexity(self) -> float:
        """
        Calculate the emotional complexity based on the distribution
        of emotional categories (entropy-based measure).
        """
        values = np.array(list(self.categories.values()))
        # Use Shannon entropy as complexity measure
        values = values[values > 0]  # Remove zeros
        if len(values) == 0:
            return 0.0
        values = values / values.sum()  # Normalize
        return -np.sum(values * np.log2(values))


class EmotionalTransformationType(Enum):
    """Types of emotional transformations that can be applied to audio."""
    AMPLIFY = auto()  # Enhance existing emotions
    DIMINISH = auto()  # Reduce existing emotions
    CONVERT = auto()  # Change emotional content to a different type
    HARMONIZE = auto()  # Create emotional harmony/balance
    CONTRAST = auto()  # Increase emotional contrast
    NARRATIVE = auto()  # Create an emotional narrative/journey
    INTENSIFY = auto()  # Make emotions more intense
    PURIFY = auto()  # Make emotions more pure/focused
    COMPLEXIFY = auto()  # Make emotions more complex/layered
    RESOLVE = auto()  # Create emotional resolution
    DESTABILIZE = auto()  # Create emotional instability/tension


@dataclass
class EmotionalTransformation:
    """
    Defines an emotional transformation to be applied to audio.
    """
    # Type of transformation
    transformation_type: EmotionalTransformationType
    
    # Target emotions (for CONVERT, HARMONIZE, etc.)
    target_emotions: Optional[List[EmotionalCategory]] = None
    
    # Target emotional dimensions (for dimensional transformations)
    target_dimensions: Optional[Dict[EmotionalDimension, float]] = None
    
    # Intensity of the transformation (0.0 to 1.0)
    intensity: float = 0.5
    
    # Parameters specific to this transformation
    parameters: Dict[str, Any] = None


class EmotionalIntelligenceModel:
    """
    Base class for emotional intelligence models used to analyze and 
    transform audio based on emotional content.
    """
    def __init__(self, model_path: str = None):
        """Initialize the emotional intelligence model."""
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the underlying machine learning model."""
        raise NotImplementedError("Subclasses must implement _load_model")
    
    def analyze_emotional_content(self, audio: np.ndarray, sr: int) -> EmotionalSignature:
        """Analyze the emotional content of audio and return an EmotionalSignature."""
        raise NotImplementedError("Subclasses must implement analyze_emotional_content")
    
    def transform_emotional_content(self, 
                                   audio: np.ndarray, 
                                   sr: int, 
                                   transformation: EmotionalTransformation) -> np.ndarray:
        """Apply an emotional transformation to audio."""
        raise NotImplementedError("Subclasses must implement transform_emotional_content")


class TensorFlowEmotionalIntelligenceModel(EmotionalIntelligenceModel):
    """
    Implementation of the EmotionalIntelligenceModel using TensorFlow.
    """
    def __init__(self, model_path: str = "models/emotional_intelligence/tensorflow"):
        """Initialize the TensorFlow-based emotional intelligence model."""
        super().__init__(model_path)
        self.feature_extractor = None
        self.dimension_model = None
        self.category_model = None
        self.temporal_model = None
    
    def _load_model(self):
        """Load TensorFlow models for emotional analysis."""
        try:
            # Load feature extraction model
            self.feature_extractor = tf.saved_model.load(
                os.path.join(self.model_path, "feature_extractor")
            )
            
            # Load dimensional emotion model
            self.dimension_model = tf.saved_model.load(
                os.path.join(self.model_path, "dimension_model")
            )
            
            # Load categorical emotion model
            self.category_model = tf.saved_model.load(
                os.path.join(self.model_path, "category_model")
            )
            
            # Load temporal emotion model
            self.temporal_model = tf.saved_model.load(
                os.path.join(self.model_path, "temporal_model")
            )
            
        except Exception as e:
            print(f"Warning: Failed to load emotional intelligence models: {e}")
            print("Using fallback algorithms for emotional processing")
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract audio features for emotional analysis."""
        if self.feature_extractor is not None:
            # Use the trained feature extractor if available
            audio_tensor = tf.convert_to_tensor(audio.reshape(1, -1), dtype=tf.float32)
            sr_tensor = tf.convert_to_tensor([sr], dtype=tf.int32)
            features = self.feature_extractor(audio_tensor, sr_tensor)
            return features.numpy()
        else:
            # Fallback to librosa features
            features = []
            
            # Mel-frequency cepstral coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            features.append(np.mean(mfcc, axis=1))
            features.append(np.std(mfcc, axis=1))
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features.append(np.mean(contrast, axis=1))
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features.append(np.mean(chroma, axis=1))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(spectral_centroid))
            features.append(np.std(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(spectral_rolloff))
            
            # Tempo and beat features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features.append(tempo)
            
            # Harmonics and perceptual features
            harmonic, percussive = librosa.effects.hpss(audio)
            features.append(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-8))
            
            # Flatten and concatenate all features
            return np.concatenate([f.flatten() if hasattr(f, 'flatten') else np.array([f]) for f in features])
    
    def analyze_emotional_content(self, audio: np.ndarray, sr: int) -> EmotionalSignature:
        """Analyze the emotional content of audio and return an EmotionalSignature."""
        # Extract features
        features = self._extract_features(audio, sr)
        
        # Initialize emotional values
        dimensions = {}
        categories = {}
        temporal_dimensions = {}
        temporal_categories = {}
        confidence = 0.7  # Default confidence
        
        if self.dimension_model is not None and self.category_model is not None:
            # Use trained models for prediction
            features_tensor = tf.convert_to_tensor(features.reshape(1, -1), dtype=tf.float32)
            
            # Predict dimensional emotions
            dimension_predictions = self.dimension_model(features_tensor).numpy()[0]
            for i, dim in enumerate(EmotionalDimension):
                if i < len(dimension_predictions):
                    dimensions[dim] = float(dimension_predictions[i])
            
            # Predict categorical emotions
            category_predictions = self.category_model(features_tensor).numpy()[0]
            for i, cat in enumerate(EmotionalCategory):
                if i < len(category_predictions):
                    categories[cat] = float(category_predictions[i])
            
            # Get temporal predictions if we have the model
            if self.temporal_model is not None:
                # Create windowed features
                hop_length = sr // 2  # 0.5 second hop
                n_windows = (len(audio) - sr) // hop_length + 1
                
                temporal_features = []
                for i in range(n_windows):
                    start = i * hop_length
                    end = start + sr
                    if end <= len(audio):
                        window_features = self._extract_features(audio[start:end], sr)
                        temporal_features.append(window_features)
                
                if temporal_features:
                    temporal_features_tensor = tf.convert_to_tensor(
                        np.array(temporal_features), dtype=tf.float32
                    )
                    
                    # Get temporal predictions
                    temporal_predictions = self.temporal_model(temporal_features_tensor).numpy()
                    
                    # Split predictions into dimensions and categories
                    dim_count = len(EmotionalDimension)
                    for i, dim in enumerate(EmotionalDimension):
                        if i < dim_count:
                            temporal_dimensions[dim] = temporal_predictions[:, i]
                    
                    for i, cat in enumerate(EmotionalCategory):
                        if i + dim_count < temporal_predictions.shape[1]:
                            temporal_categories[cat] = temporal_predictions[:, i + dim_count]
                
                confidence = 0.9  # Higher confidence with temporal analysis
        else:
            # Fallback algorithms
            
            # Simple heuristic mapping from audio features to emotional dimensions
            features_normalized = (features - np.mean(features)) / (np.std(features) + 1e-8)
            
            # Extract basic audio features for heuristic mapping
            if len(audio) > 0:
                # Energy (loudness) features
                rms = librosa.feature.rms(y=audio)[0]
                mean_energy = np.mean(rms)
                std_energy = np.std(rms)
                
                # Spectral features
                spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
                mean_centroid = np.mean(spec_centroid)
                
                # Rhythm features
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                
                # Tonal features
                chroma = librosa.feature.chroma_stft(

