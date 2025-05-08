import pytest
from src.orchestrator import BeatProduction

def test_orchestrator_init():
    bp = BeatProduction()
    assert bp is not None
    assert hasattr(bp, 'beat_generator')
