import pytest
from src.audio_engine.core import SoundGenerator

def test_sound_generator_basic():
    sg = SoundGenerator()
    result = sg.generate_tone(frequency=440, duration=1.0)
    assert result is not None
    assert hasattr(result, 'shape')
