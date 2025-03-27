"""
Core Audio Processing Engine

This module contains the AudioProcessor class which serves as the foundation
for all audio processing operations in the Enhanced Beat Empire system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

class AudioProcessor:
    """
    The core audio processing class that handles fundamental audio operations.
    
    This class provides the foundation for audio manipulation, including loading,
    processing, and exporting audio files with various enhancements.
    
    Attributes:
        sample_rate (int): The sample rate of the audio (Hz)
        channels (int): Number of audio channels (1=mono, 2=stereo)
        buffer_size (int): Size of audio processing buffer
        use_gpu (bool): Whether to use GPU acceleration if available
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, 
                 sample_rate: int = 44100, 
                 channels: int = 2,
                 buffer_size: int = 1024, 
                 use_gpu: bool = False):
        """
        Initialize the AudioProcessor with specified parameters.
        
        Args:
            sample_rate: Sample rate in Hz (default: 44100)
            channels: Number of audio channels (default: 2)
            buffer_size: Size of audio processing buffer (default: 1024)
            use_gpu: Whether to use GPU acceleration if available (default: False)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.use_gpu = use_gpu
        self.logger = logging.getLogger(__name__)
        
        self._initialize_processor()
        
    def _initialize_processor(self):
        """Initialize the audio processing engine and resources."""
        self.logger.info(f"Initializing AudioProcessor with sample rate {self.sample_rate}Hz, "
                        f"{self.channels} channels, buffer size {self.buffer_size}")
        
        # GPU initialization if requested
        if self.use_gpu:
            try:
                self.logger.info("Attempting to use GPU acceleration")
                # Here would be actual GPU initialization code
                self.gpu_available = True
            except Exception as e:
                self.logger.warning(f"GPU acceleration requested but failed: {e}")
                self.gpu_available = False
        else:
            self.gpu_available = False
    
    def generate_waveform(self, 
                         frequency: float, 
                         duration: float, 
                         waveform_type: str = 'sine',
                         amplitude: float = 0.8) -> np.ndarray:
        """
        Generate a basic waveform of specified frequency and duration.
        
        Args:
            frequency: Frequency of the waveform in Hz
            duration: Duration of the waveform in seconds
            waveform_type: Type of waveform ('sine', 'square', 'triangle', 'sawtooth')
            amplitude: Amplitude of the waveform (0.0 to 1.0)
            
        Returns:
            np.ndarray: Audio data as a numpy array
        """
        self.logger.debug(f"Generating {waveform_type} waveform at {frequency}Hz for {duration}s")
        
        # Calculate number of samples
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        
        # Generate the waveform based on type
        if waveform_type == 'sine':
            audio_data = amplitude * np.sin(2 * np.pi * frequency * t)
        elif waveform_type == 'square':
            audio_data = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == 'triangle':
            audio_data = amplitude * (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == 'sawtooth':
            audio_data = amplitude * ((2 * np.mod(frequency * t, 1)) - 1)
        else:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")
        
        # Convert to stereo if needed
        if self.channels == 2:
            audio_data = np.column_stack((audio_data, audio_data))
        
        return audio_data
    
    def apply_filter(self, 
                     audio_data: np.ndarray, 
                     filter_type: str,
                     filter_params: Dict) -> np.ndarray:
        """
        Apply a filter to the audio data.
        
        Args:
            audio_data: Input audio data as numpy array
            filter_type: Type of filter to apply ('lowpass', 'highpass', 'bandpass')
            filter_params: Parameters for the filter (cutoff frequencies, etc.)
            
        Returns:
            np.ndarray: Filtered audio data
        """
        self.logger.debug(f"Applying {filter_type} filter to audio data")
        
        # Placeholder for actual filter implementation
        # In a real implementation, we would use scipy.signal or a DSP library
        
        return audio_data  # Return unmodified for now
    
    def mix_audio_streams(self, 
                         streams: List[np.ndarray], 
                         weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Mix multiple audio streams together with optional weights.
        
        Args:
            streams: List of audio streams as numpy arrays
            weights: Optional list of weights for each stream (defaults to equal weights)
            
        Returns:
            np.ndarray: Mixed audio data
        """
        if not streams:
            raise ValueError("No audio streams provided for mixing")
            
        # Validate all streams have the same shape
        base_shape = streams[0].shape
        for i, stream in enumerate(streams):
            if stream.shape != base_shape:
                raise ValueError(f"Stream {i} shape {stream.shape} doesn't match base shape {base_shape}")
        
        # Apply weights
        if weights is None:
            weights = [1.0 / len(streams)] * len(streams)
        elif len(weights) != len(streams):
            raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of streams ({len(streams)})")
        
        # Mix the streams
        mixed = np.zeros_like(streams[0], dtype=np.float64)
        for stream, weight in zip(streams, weights):
            mixed += stream * weight
            
        # Prevent clipping
        max_value = np.max(np.abs(mixed))
        if max_value > 1.0:
            mixed = mixed / max_value
            
        return mixed
    
    def export_audio(self, 
                    audio_data: np.ndarray, 
                    filename: str,
                    file_format: str = 'wav',
                    sample_width: int = 2) -> bool:
        """
        Export audio data to a file.
        
        Args:
            audio_data: Audio data as numpy array
            filename: Output filename
            file_format: Audio file format ('wav', 'mp3', 'flac', etc.)
            sample_width: Sample width in bytes (1, 2, or 4)
            
        Returns:
            bool: True if export was successful, False otherwise
        """
        self.logger.info(f"Exporting audio to {filename} in {file_format} format")
        
        # Placeholder for actual file export
        # In a real implementation, we would use soundfile or another library
        
        return True  # Return success for now
        
    def get_processor_info(self) -> Dict:
        """
        Get information about the audio processor configuration.
        
        Returns:
            Dict: Dictionary containing processor configuration details
        """
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "buffer_size": self.buffer_size,
            "gpu_available": self.gpu_available,
            "gpu_enabled": self.use_gpu
        }

"""
Audio Processing Engine Core Module.

This module provides the core functionality for audio signal processing within the EnhancedBeatEmpire system.
It implements various techniques for audio manipulation, including:
- Frequency modulation
- Wave manipulation and synthesis
- Binaural beat generation
- GPU-accelerated processing capabilities

The module serves as the foundation for all audio processing tasks in the system, focusing on 
high-performance, precision audio manipulation for consciousness enhancement applications.

Classes:
    AudioProcessor: Base class for all audio processing operations
    WaveManipulator: Handles various wave transformations and manipulations
    FrequencyModulator: Manages frequency modulation and spectrum operations
    BinauralBeatGenerator: Creates precise binaural beat patterns
    CudaAudioProcessor: Provides GPU-accelerated audio processing capabilities

Authors: EnhancedBeatEmpire Development Team
Version: 0.1.0
"""

import os
import numpy as np
import scipy.signal as signal
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Union, Callable, Any, TypeVar, Generic, cast
import logging
from pathlib import Path
import warnings
import time
import contextlib

# Conditional imports for GPU support
try:
    import cupy as cp
    import cusignal
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA libraries not available. GPU acceleration will be disabled.")

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
AudioArray = Union[np.ndarray, "cp.ndarray"]
SampleRate = int
Seconds = float
Hertz = float
Decibels = float


class BrainwaveFrequencyRange(Enum):
    """Frequency ranges corresponding to different brainwave states."""
    
    DELTA = (0.5, 4.0)    # Deep sleep, healing
    THETA = (4.0, 8.0)    # Deep relaxation, meditation, creativity
    ALPHA = (8.0, 14.0)   # Relaxed alertness, calmness, learning
    BETA = (14.0, 30.0)   # Active thinking, focus, alertness
    GAMMA = (30.0, 100.0) # Higher cognitive processing, peak concentration
    
    def __init__(self, min_freq: Hertz, max_freq: Hertz):
        self.min_freq = min_freq
        self.max_freq = max_freq
    
    @property
    def range(self) -> Tuple[Hertz, Hertz]:
        """Get the frequency range as a tuple."""
        return (self.min_freq, self.max_freq)
    
    @property
    def center(self) -> Hertz:
        """Get the center frequency of the range."""
        return (self.min_freq + self.max_freq) / 2


@dataclass
class AudioSegment:
    """Container for audio data and its associated metadata."""
    
    data: AudioArray
    sample_rate: SampleRate
    channels: int
    duration: Seconds
    
    @property
    def num_samples(self) -> int:
        """Calculate the total number of samples in the audio data."""
        return self.data.shape[0]
    
    def __post_init__(self):
        """Validate the audio data after initialization."""
        if len(self.data.shape) > 2:
            raise ValueError(f"Audio data should be 1D or 2D, got shape {self.data.shape}")
        
        # Calculate actual duration from the data
        calculated_duration = self.num_samples / self.sample_rate
        if abs(calculated_duration - self.duration) > 0.001:
            logger.warning(
                f"Duration mismatch: specified {self.duration}s, calculated {calculated_duration}s"
            )
            self.duration = calculated_duration


class ProcessingContext:
    """Context manager for tracking and managing audio processing operations."""
    
    def __init__(self, name: str, use_gpu: bool = False):
        """
        Initialize a new processing context.
        
        Args:
            name: Name identifier for this processing context
            use_gpu: Whether to use GPU acceleration if available
        """
        self.name = name
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.start_time: float = 0
        self.metrics: Dict[str, Any] = {}
    
    def __enter__(self):
        """Enter the context, start timing."""
        self.start_time = time.time()
        logger.debug(f"Starting processing context: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, record elapsed time."""
        elapsed = time.time() - self.start_time
        self.metrics["elapsed_time"] = elapsed
        logger.debug(f"Completed processing context: {self.name} in {elapsed:.4f}s")
        
        if exc_type:
            logger.error(f"Error in processing context {self.name}: {exc_val}")
            return False
        return True
    
    def add_metric(self, key: str, value: Any):
        """Add a metric to the context."""
        self.metrics[key] = value


class AudioProcessor:
    """
    Base class for audio signal processing operations.
    
    This class provides the foundation for all audio processing capabilities,
    including GPU acceleration when available.
    """
    
    def __init__(self, sample_rate: SampleRate = 44100, use_gpu: bool = False):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate: Default sample rate for processing operations
            use_gpu: Whether to use GPU acceleration if available
        """
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        # Log initialization state
        if self.use_gpu:
            logger.info("Initializing AudioProcessor with GPU acceleration")
        else:
            logger.info("Initializing AudioProcessor with CPU processing")
    
    def to_device(self, data: np.ndarray) -> AudioArray:
        """
        Transfer numpy array to appropriate device (CPU or GPU).
        
        Args:
            data: NumPy array to transfer
            
        Returns:
            The array on the appropriate device
        """
        if self.use_gpu and CUDA_AVAILABLE:
            return cp.asarray(data)
        return data
    
    def to_cpu(self, data: AudioArray) -> np.ndarray:
        """
        Transfer array back to CPU if needed.
        
        Args:
            data: Array to transfer (could be on GPU)
            
        Returns:
            NumPy array on CPU
        """
        if self.use_gpu and CUDA_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data
    
    def process_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Process audio data with implementation-specific algorithms.
        
        Args:
            audio: Audio data to process
            
        Returns:
            Processed audio data
            
        Note:
            This is a base method intended to be overridden by subclasses.
        """
        # Base implementation just returns the original audio
        return audio
    
    def normalize(self, audio: AudioSegment, target_db: Decibels = -3.0) -> AudioSegment:
        """
        Normalize audio to a target decibel level.
        
        Args:
            audio: Audio data to normalize
            target_db: Target peak level in dB (negative value below 0dBFS)
            
        Returns:
            Normalized audio data
        """
        with ProcessingContext("normalize", self.use_gpu):
            data = self.to_device(audio.data)
            
            # Calculate current peak
            peak = np.max(np.abs(data))
            if peak == 0:
                logger.warning("Cannot normalize silent audio")
                return audio
            
            # Calculate target gain
            target_linear = 10 ** (target_db / 20.0)
            gain = target_linear / peak
            
            # Apply gain
            normalized_data = data * gain
            
            # Convert back to CPU if needed
            normalized_data = self.to_cpu(normalized_data)
            
            return AudioSegment(
                data=normalized_data,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                duration=audio.duration
            )
    
    def resample(self, audio: AudioSegment, target_rate: SampleRate) -> AudioSegment:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio: Audio data to resample
            target_rate: Target sample rate
            
        Returns:
            Resampled audio data
        """
        if audio.sample_rate == target_rate:
            return audio
            
        with ProcessingContext("resample", self.use_gpu):
            # Calculate resampling ratio
            ratio = target_rate / audio.sample_rate
            
            if self.use_gpu and CUDA_AVAILABLE:
                # Use GPU-accelerated resampling
                data_gpu = self.to_device(audio.data)
                resampled_data = cusignal.resample_poly(data_gpu, up=target_rate, down=audio.sample_rate)
                resampled_data = self.to_cpu(resampled_data)
            else:
                # Use CPU resampling
                resampled_data = signal.resample_poly(audio.data, up=target_rate, down=audio.sample_rate)
            
            # Calculate new duration
            new_duration = len(resampled_data) / target_rate
            
            return AudioSegment(
                data=resampled_data,
                sample_rate=target_rate,
                channels=audio.channels,
                duration=new_duration
            )


class WaveManipulator(AudioProcessor):
    """
    Handles various wave transformations and manipulations.
    
    This class provides methods for generating and transforming basic waveforms,
    applying filters, and performing time-domain manipulations.
    """
    
    def generate_sine_wave(
        self, 
        frequency: Hertz, 
        duration: Seconds, 
        amplitude: float = 1.0
    ) -> AudioSegment:
        """
        Generate a sine wave of specified frequency and duration.
        
        Args:
            frequency: Frequency of the sine wave in Hz
            duration: Duration of the wave in seconds
            amplitude: Peak amplitude of the wave (0.0 to 1.0)
            
        Returns:
            AudioSegment containing the generated sine wave
        """
        with ProcessingContext("generate_sine_wave", self.use_gpu):
            # Calculate number of samples
            num_samples = int(duration * self.sample_rate)
            
            # Generate time array
            t = np.linspace(0, duration, num_samples, endpoint=False)
            
            # Generate sine wave
            data = amplitude * np.sin(2 * np.pi * frequency * t)
            
            return AudioSegment(
                data=data,
                sample_rate=self.sample_rate,
                channels=1,
                duration=duration
            )
    
    def apply_envelope(
        self, 
        audio: AudioSegment, 
        attack: Seconds, 
        decay: Seconds, 
        sustain: float, 
        release: Seconds
    ) -> AudioSegment:
        """
        Apply an ADSR envelope to an audio segment.
        
        Args:
            audio: Audio data to process
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0 to 1.0)
            release: Release time in seconds
            
        Returns:
            Audio data with envelope applied
        """
        with ProcessingContext("apply_envelope", self.use_gpu):
            data = self.to_device(audio.data)
            
            # Calculate sample positions
            num_samples = len(data)
            attack_samples = int(attack * audio.sample_rate)
            decay_samples = int(decay * audio.sample_rate)
            release_samples = int(release * audio.sample_rate)
            sustain_samples = num_samples - attack_samples - decay_samples - release_samples
            
            # Ensure we have enough samples
            if sustain_samples < 0:
                logger.warning("Audio too short for specified envelope parameters")
                sustain_samples = 0
                
            # Create envelope segments
            attack_env = np.linspace(0, 1, attack_samples) if attack_samples > 0 else np.array([])
            decay_env = np.linspace(1, sustain, decay_samples) if decay_samples > 0 else np.array([])
            sustain_env = np.ones(sustain_samples) * sustain if sustain_samples > 0 else np.array([])
            release_env = np.linspace(sustain, 0, release_samples) if release_samples > 0 else np.array([])
            
            # Combine segments
            envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
            
            # Ensure envelope is the right length
            if len(envelope) < num_samples:
                envelope = np.pad(envelope, (0, num_samples - len(envelope)), 'constant', constant_values=0)
            elif len(envelope) > num_samples:
                envelope = envelope[:num_samples]
            
            # Apply envelope
            data_with_envelope = data * envelope
            
            # Convert back to CPU if needed
            data_with_envelope = self.to_cpu(data_with_envelope)
            
            return AudioSegment(
                data=data_with_envelope,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                duration=audio.duration
            )
    
    def apply_filter(
        self, 
        audio: AudioSegment, 
        filter_type: str, 
        cutoff_freq: Union[Hertz, Tuple[Hertz, Hertz]], 
        order: int = 4
    ) -> AudioSegment:
        """
        Apply various filters to an audio segment.
        
        Args:
            audio: Audio data to filter
            filter_type: Type of filter ('lowpass', 'highpass', 'bandpass', 'bandstop')
            cutoff_freq: Cutoff frequency or frequencies for the filter
            order: Filter order
            
        Returns:
            Filtered audio data
        """
        with ProcessingContext("apply_filter", self.use_gpu):
            data = audio.data
            
            # Normalize cutoff frequency to Nyquist frequency
            nyquist = audio.sample_rate / 2.0
            
            if isinstance(cutoff_freq, tuple):
                cutoff_norm = (cutoff_freq[0] / nyquist, cutoff_freq[1] / nyquist)
            else:
                cutoff_norm = cutoff_freq / nyquist
            
            # Design the filter
            b, a = signal.butter(order, cutoff_norm, filter_type)
            
            # Apply the filter
            if self.use_gpu and CUDA_AVAILABLE:
                data_gpu = self.to_device(data)
                filtered_data = cusignal.filtfilt(
                    cp.asarray(b), cp.asarray(a), data_gpu
                )
                filtered_data = self.to_cpu(filtered_data)
            else:
                filtered_data = signal.filtfilt(b, a, data)
            
            return AudioSegment(
                data=filtered_data,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                duration=audio.duration
            )
