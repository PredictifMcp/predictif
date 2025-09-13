"""Tests for the main module."""

import pytest
from predictif.main import setup_server


def test_setup_server():
    """Test that server setup runs without errors."""
    # This is a basic smoke test
    setup_server()