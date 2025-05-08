import pytest
from src.beat_generation.beat_generator import BeatGenerator

def test_beat_generator_basic():
    bg = BeatGenerator()
    beat = bg.generate_beat({'style': 'hiphop'})
    assert beat is not None
    assert 'audio' in beat
