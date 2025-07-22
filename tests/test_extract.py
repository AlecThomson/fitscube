"""Tests for extracting planes"""

from pathlib import Path

from fitscube.extract import get_output_path

def test_get_output_path() -> None:
    """Make sure the output path generated is correct"""
    
    in_fits = Path("some.example.cube.fits")
    channel_index = 10
    expected_fits = Path("some.example.cube.channel-10.fits")
    
    assert expected_fits == get_output_path(input_path=in_fits, channel_index=channel_index)
    
