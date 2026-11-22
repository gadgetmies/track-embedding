"""Step 1 of the specification: decode a file to mono 16 kHz float32.

FFmpeg decodes the first audio stream to signed 16-bit PCM at the stream's own
sample rate and in its own channel layout (no FFmpeg resampling, no FFmpeg
channel conversion). The PCM is read from FFmpeg's pipe in chunks; each chunk is
divided by 32768, averaged over its channels and fed to a streaming
libsamplerate linear resampler, so memory holds the mono 16 kHz result and one
chunk, never the whole multi-channel decode.

The resampling ratio is ``16000 / source_rate`` computed in float32, as Essentia
does (its ``Real`` type): with it the samples equal Essentia's ``MonoLoader``
bit for bit; the float64 ratio drifts by a fraction of a sample over a track.

Streaming gives exactly the samples of one ``samplerate.resample`` call over the
whole signal (tests/test_decode.py checks this at many rates and chunk sizes).
The resampler is fed blocks of 65 536 frames (the last one shorter). Avoid tiny
blocks: python-samplerate's streaming ``Resampler`` gives a different first
sample when its very first block is a single frame, and crashed (segmentation
fault) after about a million one-frame calls in a test of version 0.2.4.
"""

import subprocess
import tempfile

import numpy as np
import samplerate

from .errors import DecodeError

SAMPLE_RATE = 16000
CONVERTER = "linear"  # libsamplerate SRC_LINEAR

# Bytes read from FFmpeg per step (~6 s of 44.1 kHz stereo 16-bit PCM).
_READ_BYTES = 1 << 20
# Mono frames per resampler call.
_RESAMPLE_BLOCK = 1 << 16
_HEADER_LIMIT = 1 << 20


def resample_ratio(source_rate):
    """``16000 / source_rate`` in float32 (Essentia's ``Real``), as a float."""
    return float(np.float32(SAMPLE_RATE) / np.float32(source_rate))


class _StreamingMono16k:
    """Mono float32 blocks at ``source_rate`` in, mono float32 at 16 kHz out,
    equal to one ``samplerate.resample(whole, resample_ratio(source_rate),
    "linear")``. Input is passed to the resampler in blocks of at least
    ``block_frames`` (the last one shorter)."""

    def __init__(self, source_rate, block_frames=None):
        if source_rate <= 0:
            raise DecodeError(f"invalid source sample rate {source_rate}")
        self.ratio = resample_ratio(source_rate)
        self.block_frames = block_frames or _RESAMPLE_BLOCK
        self.passthrough = source_rate == SAMPLE_RATE
        self.resampler = None if self.passthrough else samplerate.Resampler(CONVERTER, channels=1)
        self.pending = []
        self.pending_frames = 0
        self.out = []

    def feed(self, mono):
        if not len(mono):
            return
        if self.passthrough:
            self.out.append(mono)
            return
        self.pending.append(mono)
        self.pending_frames += len(mono)
        if self.pending_frames >= self.block_frames:
            self._flush_pending()

    def _flush_pending(self):
        block = np.concatenate(self.pending) if len(self.pending) > 1 else self.pending[0]
        self.pending, self.pending_frames = [], 0
        self.out.append(self.resampler.process(block, self.ratio, end_of_input=False))

    def finish(self):
        if not self.passthrough:
            if self.pending:
                self._flush_pending()
            self.out.append(self.resampler.process(
                np.zeros(0, np.float32), self.ratio, end_of_input=True))
        pieces = [p for p in self.out if len(p)]
        if not pieces:
            return np.zeros(0, np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)


def to_mono(pcm_s16, channels):
    """Interleaved int16 frames -> float32 mono: divide by 32768, then the
    arithmetic mean of the channels (``(L + R) / 2`` for stereo)."""
    frames = pcm_s16.reshape(-1, channels).astype(np.float32) / np.float32(32768.0)
    if channels == 1:
        return frames[:, 0]
    return frames.mean(axis=1, dtype=np.float32)


def resample_to_16k(mono, source_rate, block_frames=None):
    """Resample a whole mono float32 signal to 16 kHz as the decoder does:
    streaming in blocks of ``block_frames`` (default: the decoder's block),
    which equals one ``samplerate.resample`` call over the whole signal."""
    block_frames = block_frames or _RESAMPLE_BLOCK
    stream = _StreamingMono16k(source_rate, block_frames)
    mono = np.asarray(mono, np.float32)
    for start in range(0, len(mono), block_frames):
        stream.feed(mono[start:start + block_frames])
    return stream.finish()


def _parse_wav_header(data):
    """``(channels, sample_rate, offset of the PCM)`` from the start of FFmpeg's
    streamed WAV, or ``None`` while the header is incomplete. A piped WAV's size
    fields are placeholders, so the ``data`` chunk runs to the end of the stream."""
    if len(data) < 12:
        return None
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise DecodeError("FFmpeg did not write a WAV stream")
    pos, fmt = 12, None
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        if chunk_id == b"data":
            if fmt is None:
                raise DecodeError("FFmpeg's WAV stream has no fmt chunk before its data")
            return fmt[0], fmt[1], pos + 8
        if pos + 8 + size > len(data):
            return None
        if chunk_id == b"fmt ":
            body = data[pos + 8:pos + 8 + size]
            channels = int.from_bytes(body[2:4], "little")
            rate = int.from_bytes(body[4:8], "little")
            bits = int.from_bytes(body[14:16], "little")
            if bits != 16 or channels < 1:
                raise DecodeError(
                    f"FFmpeg wrote {bits}-bit PCM with {channels} channels; expected 16-bit")
            fmt = (channels, rate)
        pos += 8 + size + (size & 1)
    return None


def ffmpeg_command(path, ffmpeg="ffmpeg"):
    """The FFmpeg command of step 1: first audio stream, 16-bit PCM, the
    stream's own rate and layout, streamed as WAV to standard output."""
    return [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
            "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "wav", "-bitexact", "-"]


def decode(path, ffmpeg="ffmpeg"):
    """Decode ``path`` to mono float32 samples at 16 kHz (specification step 1).

    ``ffmpeg`` is the FFmpeg executable (a name on ``PATH`` or a path). Raises
    :class:`DecodeError` with FFmpeg's own error output when FFmpeg cannot be
    started or exits with an error (no audio stream, unreadable file, ...).
    Audio of zero length decodes to an empty array."""
    command = ffmpeg_command(path, ffmpeg)
    with tempfile.TemporaryFile() as stderr:
        try:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=stderr)
        except OSError as error:
            raise DecodeError(f"could not start FFmpeg ({ffmpeg!r}): {error}") from error
        stream = None
        failure = None
        try:
            stream, failure = _read_pcm(proc.stdout)
        finally:
            proc.stdout.close()
            returncode = proc.wait()
        if returncode != 0 or failure is not None:
            stderr.seek(0)
            message = stderr.read().decode(errors="replace").strip()
            reason = f"exit status {returncode}" if returncode else failure
            raise DecodeError(f"FFmpeg could not decode {str(path)!r} ({reason}): "
                              f"{message or '(no error output)'}")
    if stream is None:
        return np.zeros(0, np.float32)
    return stream.finish()


def _read_pcm(stdout):
    """Read FFmpeg's WAV from ``stdout``; returns ``(stream, failure)``. A
    failure is reported with FFmpeg's stderr by the caller (after the process
    has exited), so the message names FFmpeg's reason, not only ours."""
    buf = b""
    while True:
        header = _parse_wav_header(buf)
        if header is not None:
            break
        block = stdout.read(4096)
        if not block:
            return None, ("no output" if not buf else "incomplete WAV header")
        buf += block
        if len(buf) > _HEADER_LIMIT:
            return None, "WAV header too long"
    channels, rate, offset = header
    stream = _StreamingMono16k(rate)
    frame_bytes = 2 * channels
    rest = buf[offset:]
    while True:
        block = stdout.read(_READ_BYTES)
        data = rest + block if block else rest
        usable = len(data) // frame_bytes * frame_bytes
        if usable:
            stream.feed(to_mono(np.frombuffer(data[:usable], dtype="<i2"), channels))
        rest = data[usable:]
        if not block:
            break
    if rest:
        return stream, f"{len(rest)} trailing bytes are not a whole frame"
    return stream, None
