import numpy as np
import librosa
import soundfile as sf
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import scipy.signal as signal

logger = logging.getLogger(__name__)

@dataclass
class SoundParams:
    frequency: float
    duration: float
    sample_rate: int
    waveform: str
    amplitude: float
    phase: float

class SoundGenerator:
    """
    Advanced sound generation system with support for multiple waveforms,
    frequency modulation, and sacred geometry integration.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize sound generation parameters"""
        self.params = {
            'waveforms': ['sine', 'square', 'sawtooth', 'triangle'],
            'base_frequency': 432,  # Hz
            'duration_range': (0.1, 4.0),  # seconds
            'amplitude_range': (0.0, 1.0),
            'frequency_ratios': [1.0, 1.5, 2.0, 2.5, 3.0],  # Harmonic series
            'default_duration': 0.5,  # seconds
            'default_amplitude': 0.8
        }
        
    def generate_beat(self, style: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a beat based on style parameters.
        
        Args:
            style: Dictionary containing style parameters
            
        Returns:
            Dictionary containing generated audio and parameters
        """
        try:
            # Extract style parameters
            tempo = style.get('tempo', 120)
            rhythm_complexity = style.get('rhythm_complexity', 0.5)
            base_frequency = style.get('base_frequency', self.params['base_frequency'])
            
            # Calculate rhythm pattern
            pattern = self._generate_rhythm_pattern(
                tempo,
                rhythm_complexity
            )
            
            # Generate individual sounds
            sounds = []
            for hit in pattern:
                sound_params = SoundParams(
                    frequency=base_frequency * hit['frequency_ratio'],
                    duration=hit['duration'],
                    sample_rate=self.sample_rate,
                    waveform=hit['waveform'],
                    amplitude=hit['amplitude'],
                    phase=hit['phase']
                )
                sounds.append(self._generate_sound(sound_params))
            
            # Combine sounds into final beat
            beat = self._combine_sounds(sounds, pattern)
            
            return {
                'audio': beat,
                'sample_rate': self.sample_rate,
                'parameters': {
                    'tempo': tempo,
                    'rhythm_complexity': rhythm_complexity,
                    'base_frequency': base_frequency,
                    'pattern': pattern
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating beat: {str(e)}")
            raise
            
    def _generate_rhythm_pattern(self, tempo: float,
                              complexity: float) -> List[Dict[str, Any]]:
        """Generate rhythm pattern based on tempo and complexity"""
        # Calculate basic timing parameters
        beat_duration = 60.0 / tempo  # seconds per beat
        subdivision = 4 * int(1 + complexity * 3)  # 16th notes to 64th notes
        step_duration = beat_duration / subdivision
        
        pattern = []
        num_beats = 4  # One bar
        
        for beat in range(num_beats * subdivision):
            # Determine if there should be a hit at this step
            if self._should_generate_hit(beat, subdivision, complexity):
                hit = {
                    'time': beat * step_duration,
                    'duration': self._generate_duration(complexity),
                    'frequency_ratio': self._select_frequency_ratio(complexity),
                    'waveform': self._select_waveform(complexity),
                    'amplitude': self._generate_amplitude(beat, subdivision),
                    'phase': self._generate_phase(complexity)
                }
                pattern.append(hit)
                
        return pattern
        
    def _should_generate_hit(self, step: int,
                          subdivision: int,
                          complexity: float) -> bool:
        """Determine if a hit should be generated at this step"""
        # Basic rhythm probability
        if step % (subdivision // 4) == 0:  # Quarter notes
            return True
            
        # Additional hits based on complexity
        probability = complexity * 0.5  # 0-0.5 chance for additional hits
        return np.random.random() < probability
        
    def _generate_duration(self, complexity: float) -> float:
        """Generate sound duration based on complexity"""
        min_dur, max_dur = self.params['duration_range']
        base_duration = self.params['default_duration']
        
        # Vary duration based on complexity
        variation = (max_dur - min_dur) * complexity * np.random.random()
        return base_duration + variation
        
    def _select_frequency_ratio(self, complexity: float) -> float:
        """Select frequency ratio based on complexity"""
        available_ratios = self.params['frequency_ratios']
        
        if complexity < 0.3:
            # Simple ratios for low complexity
            return available_ratios[0]
        elif complexity < 0.7:
            # Medium complexity - first three ratios
            return np.random.choice(available_ratios[:3])
        else:
            # High complexity - all ratios
            return np.random.choice(available_ratios)
            
    def _select_waveform(self, complexity: float) -> str:
        """Select waveform based on complexity"""
        available_waveforms = self.params['waveforms']
        
        if complexity < 0.3:
            # Simple waveforms for low complexity
            return 'sine'
        elif complexity < 0.7:
            # Medium complexity - sine and triangle
            return np.random.choice(['sine', 'triangle'])
        else:
            # High complexity - all waveforms
            return np.random.choice(available_waveforms)
            
    def _generate_amplitude(self, step: int, subdivision: int) -> float:
        """Generate amplitude based on step position"""
        min_amp, max_amp = self.params['amplitude_range']
        base_amplitude = self.params['default_amplitude']
        
        # Emphasize beats
        if step % subdivision == 0:
            return max_amp
        elif step % (subdivision // 4) == 0:
            return base_amplitude
        else:
            return min_amp + (max_amp - min_amp) * np.random.random()
            
    def _generate_phase(self, complexity: float) -> float:
        """Generate phase offset based on complexity"""
        return 2 * np.pi * complexity * np.random.random()
        
    def _generate_sound(self, params: SoundParams) -> np.ndarray:
        """Generate individual sound based on parameters"""
        # Generate time array
        t = np.linspace(0, params.duration, int(params.duration * params.sample_rate))
        
        # Generate waveform
        if params.waveform == 'sine':
            wave = np.sin(2 * np.pi * params.frequency * t + params.phase)
        elif params.waveform == 'square':
            wave = signal.square(2 * np.pi * params.frequency * t + params.phase)
        elif params.waveform == 'sawtooth':
            wave = signal.sawtooth(2 * np.pi * params.frequency * t + params.phase)
        elif params.waveform == 'triangle':
            wave = signal.sawtooth(2 * np.pi * params.frequency * t + params.phase, width=0.5)
        else:
            raise ValueError(f"Unknown waveform: {params.waveform}")
            
        # Apply amplitude
        wave *= params.amplitude
        
        # Apply envelope
        envelope = self._create_envelope(len(wave))
        wave *= envelope
        
        return wave
        
    def _combine_sounds(self, sounds: List[np.ndarray],
                      pattern: List[Dict[str, Any]]) -> np.ndarray:
        """Combine individual sounds into final beat"""
        # Calculate total duration
        max_time = max(hit['time'] + hit['duration'] for hit in pattern)
        total_samples = int(max_time * self.sample_rate)
        
        # Create output array
        beat = np.zeros(total_samples)
        
        # Add each sound at its time position
        for sound, hit in zip(sounds, pattern):
            start_sample = int(hit['time'] * self.sample_rate)
            end_sample = start_sample + len(sound)
            beat[start_sample:end_sample] += sound
            
        # Normalize
        beat = beat / np.max(np.abs(beat))
        
        return beat
        
    def _create_envelope(self, length: int) -> np.ndarray:
        """Create ADSR envelope"""
        # Define envelope parameters
        attack = int(0.1 * length)
        decay = int(0.2 * length)
        sustain = int(0.5 * length)
        release = length - attack - decay - sustain
        
        # Create envelope segments
        attack_env = np.linspace(0, 1, attack)
        decay_env = np.linspace(1, 0.7, decay)
        sustain_env = np.ones(sustain) * 0.7
        release_env = np.linspace(0.7, 0, release)
        
        # Combine segments
        envelope = np.concatenate([
            attack_env,
            decay_env,
            sustain_env,
            release_env
        ])
        
        return envelope
        
    def save_audio(self, audio: np.ndarray, file_path: str):
        """Save audio to file"""
        try:
            sf.write(file_path, audio, self.sample_rate)
            logger.info(f"Saved audio to {file_path}")
        except Exception as e:
            logger.error(f"Error saving audio: {str(e)}")
            raise