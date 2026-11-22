"""Steps 2-3 without the model: frame and patch counts, and the front-end
against a frame-by-frame computation."""

import numpy as np

from track_embedding.embedding import MelFrontend, frame_count, patch_count, patches


def test_frame_count_is_ceil_n_over_256_plus_one():
    assert [frame_count(n) for n in (0, 1, 256, 257, 512)] == [0, 2, 2, 3, 3]


def test_patches_are_whole_128_frame_patches_every_62_frames():
    assert [patch_count(n) for n in (0, 127, 128, 189, 190, 251, 252)] == [0, 0, 1, 1, 2, 2, 3]


def test_one_patch_needs_more_than_32256_samples():
    assert patch_count(frame_count(32256)) == 0
    assert patch_count(frame_count(32257)) == 1


def test_frontend_matches_a_frame_by_frame_computation():
    frontend = MelFrontend()
    rng = np.random.default_rng(3)
    samples = (rng.standard_normal(5000) * 0.2).astype(np.float32)
    bands = frontend(samples)
    assert bands.shape == (frame_count(len(samples)), 96)
    padded = np.concatenate([np.zeros(256), samples.astype(np.float64), np.zeros(512)])
    for k in (0, 1, 9, len(bands) - 1):
        frame = padded[k * 256:k * 256 + 512] * frontend.window
        power = np.abs(np.fft.rfft(frame)) ** 2
        expected = np.log10(1 + 10000 * (frontend.mel.astype(np.float64) @ power))
        np.testing.assert_allclose(bands[k], expected, rtol=1e-4, atol=1e-4)


def test_patches_are_frame_windows():
    mel = np.arange(300 * 96, dtype=np.float32).reshape(300, 96)
    got = patches(mel)
    assert got.shape == (patch_count(300), 128, 96)
    assert np.array_equal(got[2], mel[124:252])
