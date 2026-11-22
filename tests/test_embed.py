"""Steps 2-5 with the model (needs TRACK_EMBEDDING_MODELS_DIR)."""

import numpy as np
import pytest

from track_embedding import AudioTooShortError, Embedder, embed
from track_embedding.embedding import frame_count, patch_count


@pytest.fixture(scope="module")
def embedder(models_dir):
    return Embedder(models_dir)


def _signal(n, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 2 * t))
    return (tone + 0.05 * rng.standard_normal(n)).astype(np.float32)


def test_vector_is_1280_float32_and_the_mean_of_the_patches(embedder, models_dir):
    samples = _signal(16000 * 6)
    vector = embed(samples, models_dir)
    assert vector.shape == (1280,) and vector.dtype == np.float32
    rows = embedder.patch_embeddings(samples)
    assert rows.shape == (patch_count(frame_count(len(samples))), 1280) == (5, 1280)
    np.testing.assert_allclose(vector, rows.mean(axis=0), rtol=0, atol=1e-6)


def test_shorter_than_one_patch_has_no_embedding(embedder):
    with pytest.raises(AudioTooShortError, match="one patch needs 128"):
        embedder(_signal(32256))
    assert embedder.patch_embeddings(_signal(32257)).shape == (1, 1280)


def test_batches_of_64_with_padding_discarded(embedder):
    # 70 patches: one full batch and one padded batch of 6. The first 5 patches
    # alone (a padded batch of 5) give the same rows.
    long = _signal(256 * (128 + 69 * 62) + 1000, seed=1)
    rows = embedder.patch_embeddings(long)
    assert rows.shape == (70, 1280)
    prefix = long[:256 * (128 + 4 * 62) + 512]  # frames 0..375 see the same samples
    head = embedder.patch_embeddings(prefix)
    assert head.shape == (5, 1280)
    np.testing.assert_allclose(head, rows[:5], rtol=0, atol=1e-5)
