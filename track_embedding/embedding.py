"""Steps 2-5 of the specification: log-mel front-end, patches, the
Discogs-EffNet model in ONNX form, and the mean over patches.

The front-end is Essentia's ``TensorflowInputMusiCNN`` in numpy: 512-sample
frames every 256 samples (256 zeros in front, 512 behind, ``ceil(n / 256) + 1``
frames), the window and 96 x 257 mel matrix from ``constants/mel_frontend.npz``,
the power spectrum and ``log10(1 + 10000 x)``.
"""

import functools
import os

import numpy as np

from . import manifest
from .decoding import SAMPLE_RATE, decode
from .errors import AudioTooShortError

FRAME_SIZE = 512
HOP_SIZE = 256
PATCH_FRAMES = 128
PATCH_HOP = 62
BATCH = 64  # the published graph takes exactly 64 patches
DIMENSIONS = 1280
OUTPUT = "embeddings"


class MelFrontend:
    """Mono 16 kHz float32 samples -> ``(frames, 96)`` log-mel bands."""

    # Frames per FFT block: bounds memory for long audio.
    BLOCK = 8192

    def __init__(self):
        consts = np.load(manifest.verified_path(manifest.CONSTANTS_DIR, manifest.MEL_FRONTEND))
        self.window = consts["window"].astype(np.float32)  # (512,)
        self.mel = consts["mel"].astype(np.float32)        # (96, 257)

    def __call__(self, samples):
        from scipy.fft import rfft  # keeps float32 -> complex64 (numpy upcasts)
        samples = np.asarray(samples, dtype=np.float32)
        n_frames = frame_count(len(samples))
        padded = np.concatenate([np.zeros(HOP_SIZE, np.float32), samples,
                                 np.zeros(FRAME_SIZE, np.float32)])
        frames = np.lib.stride_tricks.sliding_window_view(
            padded, FRAME_SIZE)[::HOP_SIZE][:n_frames]
        out = np.empty((n_frames, self.mel.shape[0]), np.float32)
        for start in range(0, n_frames, self.BLOCK):
            block = frames[start:start + self.BLOCK] * self.window
            spectrum = rfft(block, axis=1)
            power = spectrum.real ** 2 + spectrum.imag ** 2
            out[start:start + len(block)] = np.log10(1.0 + 10000.0 * (power @ self.mel.T))
        return out


def frame_count(n_samples):
    """Front-end frames of ``n_samples``: ``ceil(n / 256) + 1`` (none for none)."""
    return -(-n_samples // HOP_SIZE) + 1 if n_samples else 0


def patch_count(n_frames):
    """Patches of 128 frames every 62 that end before the last frame, as
    Essentia counts them: ``ceil((n_frames - 128) / 62)``, none for 128 frames
    or fewer."""
    return 0 if n_frames <= PATCH_FRAMES else -(-(n_frames - PATCH_FRAMES) // PATCH_HOP)


def patches(mel, start=0, stop=None):
    """Patches ``[start, stop)`` of ``(frames, 96)`` bands as a contiguous
    ``(n, 128, 96)`` array; only the requested patches are materialised."""
    count = patch_count(len(mel))
    stop = count if stop is None else min(stop, count)
    view = np.lib.stride_tricks.sliding_window_view(mel, PATCH_FRAMES, axis=0)
    return np.ascontiguousarray(
        view[start * PATCH_HOP:stop * PATCH_HOP:PATCH_HOP].transpose(0, 2, 1))


class Embedder:
    """The model loaded once: call it with mono 16 kHz samples to get the
    1280-wide float32 embedding. ``threads`` caps ONNX Runtime's threads
    (``None`` = its default)."""

    def __init__(self, models_dir, threads=None):
        import onnxruntime as ort
        path = manifest.verified_path(models_dir, manifest.EFFNET_MODEL)
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.frontend = MelFrontend()

    def patch_embeddings(self, samples):
        """``(patches, 1280)`` per-patch embeddings, run in batches of 64 with
        the last batch zero-padded and its padding rows discarded."""
        mel = self.frontend(samples)
        total = patch_count(len(mel))
        if total == 0:
            raise AudioTooShortError(
                f"audio of {len(samples)} samples ({len(samples) / SAMPLE_RATE:.3f} s at "
                f"16 kHz) gives {len(mel)} front-end frames; one patch needs "
                f"more than {PATCH_FRAMES} (more than {(PATCH_FRAMES - 1) * HOP_SIZE} samples)")
        out = np.empty((total, DIMENSIONS), np.float32)
        for start in range(0, total, BATCH):
            chunk = patches(mel, start, start + BATCH)
            n = len(chunk)
            if n < BATCH:
                chunk = np.concatenate(
                    [chunk, np.zeros((BATCH - n,) + chunk.shape[1:], chunk.dtype)])
            out[start:start + n] = self.session.run([OUTPUT], {self.input: chunk})[0][:n]
        return out

    def __call__(self, samples):
        rows = self.patch_embeddings(samples)
        return rows.mean(axis=0, dtype=np.float64).astype(np.float32)


@functools.lru_cache(maxsize=4)
def _embedder(models_dir):
    return Embedder(models_dir)


def embed(samples, models_dir):
    """The ``track-embedding/1`` vector (float32, 1280, not normalised) of mono
    16 kHz float32 ``samples``. The model is loaded from ``models_dir`` once per
    process and verified against its pinned sha256. Raises
    :class:`AudioTooShortError` when the audio is shorter than one patch."""
    return _embedder(os.path.abspath(os.fspath(models_dir)))(samples)


def embed_file(path, models_dir, ffmpeg="ffmpeg"):
    """:func:`decode` then :func:`embed`."""
    return embed(decode(path, ffmpeg=ffmpeg), models_dir)
