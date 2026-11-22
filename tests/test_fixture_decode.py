"""Step 1 on the conformance fixtures: the streaming decoder gives exactly the
samples of the whole-file pipeline (one FFmpeg run, one resample call)."""

import os
import subprocess

import numpy as np
import pytest
import samplerate

from track_embedding import decode
from track_embedding import decoding as decode_module
from track_embedding.decoding import resample_ratio, to_mono

from conftest import FIXTURES_DIR, fixture_listing


def _one_shot(path):
    """Step 1 without streaming: the whole decode in memory, one resample call."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=sample_rate,channels", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    rate, channels = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0",
         "-c:a", "pcm_s16le", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    mono = to_mono(np.frombuffer(raw, "<i2"), channels)
    if rate == 16000:
        return mono
    return samplerate.resample(mono, resample_ratio(rate), "linear")


@pytest.mark.parametrize("fixture", [f["audio"] for f in fixture_listing()])
def test_decode_equals_the_one_shot_pipeline(fixture, ffmpeg, monkeypatch):
    path = os.path.join(FIXTURES_DIR, fixture)
    expected = _one_shot(path)
    assert np.array_equal(decode(path), expected)
    # Small, odd pipe reads and resampler blocks: frames split across reads.
    monkeypatch.setattr(decode_module, "_READ_BYTES", 4099)
    monkeypatch.setattr(decode_module, "_RESAMPLE_BLOCK", 1001)
    assert np.array_equal(decode(path), expected)
