"""Conformance with track-embedding/1: every fixture's vector against the
reference vector made by fomoplayer's Essentia pipeline (needs
TRACK_EMBEDDING_MODELS_DIR)."""

import hashlib
import os

import numpy as np
import pytest

from track_embedding import embed_file

from conftest import FIXTURES_DIR, fixture_listing

TOLERANCE = 0.99999  # SPECIFICATION.md, "Conformance"


def test_the_fixtures_cover_the_specifications_formats():
    assert [f["audio"].rsplit(".", 1)[1] for f in fixture_listing()] == [
        "mp3", "m4a", "wav", "flac", "wav"]


@pytest.mark.parametrize("fixture", fixture_listing(), ids=lambda f: f["audio"])
def test_vector_matches_the_reference(fixture, models_dir, ffmpeg):
    audio = os.path.join(FIXTURES_DIR, fixture["audio"])
    with open(audio, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == fixture["audio_sha256"]
    reference = np.load(os.path.join(FIXTURES_DIR, fixture["reference"]))
    vector = embed_file(audio, models_dir)
    assert vector.shape == reference.shape == (1280,)
    cosine = float(np.dot(vector, reference) /
                   (np.linalg.norm(vector) * np.linalg.norm(reference)))
    assert cosine >= TOLERANCE, f"{fixture['audio']}: cosine {cosine:.9f} < {TOLERANCE}"
