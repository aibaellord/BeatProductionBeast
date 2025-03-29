"""
BeatProductionBeast: AI-Powered Music Production Toolkit

BeatProductionBeast is a comprehensive toolkit for creating, analyzing, and enhancing
musical beats and compositions using neural network techniques. This package brings together
various components for audio processing, neural beat generation, pattern recognition,
and harmonic enhancement.

Quick Start:
------------
    import beatproductionbeast as bpb
    
    # Generate a new beat
    beat = bpb.generate_beat(style="hip-hop", bpm=90)
    
    # Apply neural processing to enhance the beat
    enhanced_beat = bpb.apply_neural_processing(beat, intensity=0.7)
    
    # Export the result
    enhanced_beat.export("my_beat.wav")

Main Components:
---------------
- Neural Beat Architect: Creates original beat patterns based on learned styles
- Audio Engine: Processes and manipulates audio signals and waveforms
- Neural Processing: Applies neural network models to audio data
- Pattern Recognition: Identifies patterns in existing music
- Harmonic Enhancement: Improves harmonic content of audio
- Style Analysis: Analyzes and categorizes musical styles
- Fusion Generator: Combines different musical elements and styles

See individual module documentation for detailed usage information.
"""

__version__ = "0.1.0"

# Neural Beat Architect
from .neural_beat_architect import (
    BeatGenerator,
    StyleModel,
    GrooveAnalyzer,
    RhythmPattern
)

# Audio Engine
from .audio_engine import (
    AudioProcessor,
    SoundGenerator,
    MixerInterface,
    WaveformAnalyzer,
    AudioEffect
)

# Neural Processing
from .neural_processing import (
    ModelLoader,
    Predictor,
    FeatureExtractor,
    NeuralModel
)

# Beat Generation
from .beat_generation import (
    BeatMaker,
    DrumSequencer,
    LoopGenerator,
    PatternLibrary
)

# Fusion Generator
from .fusion_generator import (
    StyleFusion,
    GenreMerger,
    CrossGenreAdapter,
    FusionMatrix
)

# Harmonic Enhancement
from .harmonic_enhancement import (
    HarmonicAnalyzer,
    ChordEnhancer,
    ScaleDetector,
    TonalAdjuster
)

# Pattern Recognition
from .pattern_recognition import (
    PatternMatcher,
    RhythmicAnalyzer,
    PatternDatabase,
    SignatureDetector
)

# Style Analysis
from .style_analysis import (
    StyleClassifier,
    GenreIdentifier,
    FeatureExtraction,
    StyleMapping
)

# Utils
from .utils import (
    audio_conversion,
    file_handling,
    visualization,
    midi_tools,
    config
)

# Convenience functions
def generate_beat(style="default", bpm=120, length=8):
    """
    Generate a new beat with specified style and parameters.
    
    Args:
        style (str): Musical style/genre for the beat
        bpm (int): Beats per minute
        length (int): Length in bars
        
    Returns:
        A generated beat object
    """
    generator = BeatGenerator()
    return generator.create(style=style, bpm=bpm, length=length)

def apply_neural_processing(audio_data, model="default", intensity=0.5):
    """
    Apply neural processing to enhance audio data.
    
    Args:
        audio_data: Input audio data
        model (str): Name of the model to use
        intensity (float): Processing intensity (0.0-1.0)
        
    Returns:
        Processed audio data
    """
    processor = NeuralModel(model)
    return processor.process(audio_data, intensity=intensity)

def analyze_style(audio_data):
    """
    Analyze the musical style of the provided audio.
    
    Args:
        audio_data: Input audio data
        
    Returns:
        Style analysis results
    """
    analyzer = StyleClassifier()
    return analyzer.classify(audio_data)

def enhance_harmonics(audio_data, strength=0.6):
    """
    Enhance the harmonic content of audio data.
    
    Args:
        audio_data: Input audio data
        strength (float): Enhancement strength (0.0-1.0)
        
    Returns:
        Harmonically enhanced audio data
    """
    enhancer = HarmonicAnalyzer()
    return enhancer.enhance(audio_data, strength=strength)

"""
BeatProductionBeast - Neural Music Production Tool

This package provides a comprehensive set of tools for AI-assisted music and beat production,
combining neural networks with audio processing capabilities.
"""

__version__ = '0.1.0'

# Import main modules for easy access
try:
    # Neural Beat Architect module
    from .neural_beat_architect import (
        BeatGenerator,
        PatternCreator,
        RhythmAnalyzer,
    )
    
    # Audio Engine module
    from .audio_engine import (
        AudioProcessor,
        SoundGenerator,
        MixerInterface,
    )
    
    # Neural Processing module
    from .neural_processing import (
        ModelLoader,
        Predictor,
        FeatureExtractor,
    )
    
    # Add other commonly used components here
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")

# Define what is available when using "from beatproductionbeast import *"
__all__ = [
    # Neural Beat Architect components
    'BeatGenerator',
    'PatternCreator',
    'RhythmAnalyzer',
    
    # Audio Engine components
    'AudioProcessor',
    'SoundGenerator',
    'MixerInterface',
    
    # Neural Processing components
    'ModelLoader',
    'Predictor',
    'FeatureExtractor',
    
    # Add other exports here
]

