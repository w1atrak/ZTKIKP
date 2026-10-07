"""Smoke tests for repository setup (phase W0)."""

from ppspam import __version__
from ppspam.config import SEED


def test_version():
    assert __version__ == "0.1.0"


def test_seed():
    assert SEED == 42
