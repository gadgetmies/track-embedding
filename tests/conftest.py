"""Shared test fixtures.

Tests that run the model take the folder holding the downloaded model files from
the environment variable ``TRACK_EMBEDDING_MODELS_DIR``. When it is unset they
fail (never skip), so a run without the models cannot pass as green.
"""

import json
import os
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(ROOT, "fixtures", "track-embedding-1")
MODELS_ENV = "TRACK_EMBEDDING_MODELS_DIR"


def fixture_listing():
    with open(os.path.join(FIXTURES_DIR, "fixtures.json"), encoding="utf-8") as handle:
        return json.load(handle)["fixtures"]


@pytest.fixture(scope="session")
def models_dir():
    path = os.environ.get(MODELS_ENV)
    if not path:
        pytest.fail(
            f"{MODELS_ENV} is not set. This test runs the model: download it once with\n"
            f"  python -c \"import track_embedding; track_embedding.download_models("
            f"'models', licence_accepted=True)\"\n"
            f"(only after accepting the models' licence, CC BY-NC-SA 4.0) and run\n"
            f"  {MODELS_ENV}=models python -m pytest", pytrace=False)
    if not os.path.isdir(path):
        pytest.fail(f"{MODELS_ENV}={path!r} is not a folder", pytrace=False)
    return path


@pytest.fixture(scope="session")
def ffmpeg():
    """The FFmpeg on PATH (the package's default); fails clearly when missing."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.fail(f"FFmpeg is needed on PATH for this test: {error}", pytrace=False)
    return "ffmpeg"
