"""Step 1 (decode): streaming equals one-shot, channel mean, no resampling at
16 kHz, first audio stream, and errors that carry FFmpeg's message. The same
equality on the fixture files is in test_fixture_decode.py."""

import subprocess
import wave

import numpy as np
import pytest
import samplerate

from track_embedding import DecodeError, decode
from track_embedding.decoding import resample_ratio, resample_to_16k

RATES = [8000, 11025, 12000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000]


@pytest.mark.parametrize("rate", RATES)
def test_streaming_resample_equals_one_call_over_the_whole_signal(rate):
    rng = np.random.default_rng(rate)
    signal = (rng.standard_normal(rate * 7) * 0.3).astype(np.float32)
    whole = samplerate.resample(signal, resample_ratio(rate), "linear")
    for block in (2, 3, 997, 4096, 65536, len(signal)):
        assert np.array_equal(resample_to_16k(signal, rate, block), whole), block


def test_ratio_is_single_precision_like_essentia():
    assert resample_ratio(44100) == float(np.float32(16000) / np.float32(44100))
    assert resample_ratio(44100) != 16000 / 44100
    assert resample_ratio(16000) == 1.0


def _write_wav(path, rate, frames):
    frames = np.asarray(frames, dtype="<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(frames.shape[1])
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(frames.tobytes())


def test_stereo_is_the_mean_of_the_channels_not_the_sum(tmp_path, ffmpeg):
    rng = np.random.default_rng(0)
    frames = rng.integers(-30000, 30000, size=(16000, 2))
    _write_wav(tmp_path / "s.wav", 16000, frames)
    expected = ((frames[:, 0] / 32768.0 + frames[:, 1] / 32768.0) / 2).astype(np.float32)
    assert np.array_equal(decode(tmp_path / "s.wav"), expected)


def test_sixteen_kilohertz_mono_is_not_resampled(tmp_path, ffmpeg):
    frames = np.arange(-8000, 8000, dtype=np.int64).reshape(-1, 1) * 3
    _write_wav(tmp_path / "m.wav", 16000, frames)
    assert np.array_equal(decode(tmp_path / "m.wav"),
                          (frames[:, 0] / 32768.0).astype(np.float32))


def test_the_first_audio_stream_is_decoded(tmp_path, ffmpeg):
    # Stream 0: a mono tone. Stream 1: stereo silence, which FFmpeg's own
    # default stream choice (most channels) would pick.
    path = tmp_path / "two.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:r=16000:d=3",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=stereo", "-t", "3",
         "-map", "0:a", "-map", "1:a", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, check=True)
    samples = decode(path)
    assert len(samples) == 48000
    assert np.abs(samples).max() > 0.1


def test_unreadable_file_raises_with_ffmpegs_message(tmp_path, ffmpeg):
    path = tmp_path / "not-audio.mp3"
    path.write_bytes(b"this is not audio" * 100)
    with pytest.raises(DecodeError) as caught:
        decode(path)
    message = str(caught.value)
    assert "not-audio.mp3" in message
    assert "(no error output)" not in message
    assert len(message.split("): ", 1)[1]) > 10  # FFmpeg's own text follows


def test_file_without_audio_raises_with_ffmpegs_message(tmp_path, ffmpeg):
    path = tmp_path / "picture.png"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=16x16:d=1", "-frames:v", "1", str(path)],
                   capture_output=True, check=True)
    with pytest.raises(DecodeError, match="matches no streams"):
        decode(path)


def test_missing_ffmpeg_is_a_clear_error(tmp_path):
    _write_wav(tmp_path / "m.wav", 16000, np.zeros((16000, 1)))
    with pytest.raises(DecodeError, match="could not start FFmpeg"):
        decode(tmp_path / "m.wav", ffmpeg=str(tmp_path / "no-ffmpeg-here"))
