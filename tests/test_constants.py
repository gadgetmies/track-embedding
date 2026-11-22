"""constants/MANIFEST pins every file; the package refuses another file."""

import os
import shutil

import pytest

import track_embedding
from track_embedding import Embedder, ModelHashMismatchError, ModelsNotFoundError, manifest


def test_version_identifier():
    assert track_embedding.VERSION == "track-embedding/1"


def test_manifest_pins_the_front_end_and_the_model():
    files = manifest.FILES
    assert files[manifest.MEL_FRONTEND].bundled
    model = files[manifest.EFFNET_MODEL]
    assert model.source.startswith("https://essentia.upf.edu/models/")
    assert model.source.endswith("/" + manifest.EFFNET_MODEL)
    assert {f.name for f in manifest.model_files()} == {
        manifest.EFFNET_MODEL, "discogs_multi_embeddings-effnet-bs64-1.json"}


def test_bundled_front_end_constants_match_the_manifest():
    path = manifest.verified_path(manifest.CONSTANTS_DIR, manifest.MEL_FRONTEND)
    assert os.path.getsize(path) == manifest.FILES[manifest.MEL_FRONTEND].size


def test_missing_model_names_the_download(tmp_path):
    with pytest.raises(ModelsNotFoundError, match="download_models"):
        Embedder(tmp_path)


def test_another_model_file_is_refused(tmp_path):
    shutil.copy(os.path.join(manifest.CONSTANTS_DIR, manifest.MEL_FRONTEND),
                tmp_path / manifest.EFFNET_MODEL)
    with pytest.raises(ModelHashMismatchError, match="another file"):
        Embedder(tmp_path)
