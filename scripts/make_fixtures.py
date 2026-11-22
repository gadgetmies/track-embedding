"""Generate the synthetic conformance audio of fixtures/track-embedding-1/.

Each fixture is a short, deterministic, music-like loop (kick, noise-burst snare
and hi-hats, a bass line, a chord pad; stereo files pan the parts differently
left and right) synthesised in numpy at 48 kHz and encoded by FFmpeg in one of
the formats the specification's conformance section lists.

The committed audio files are the fixtures; this script documents how they were
made. Running it again with another FFmpeg or numpy build may produce other
bytes, and then the reference vectors must be made again too
(scripts/make_reference_vectors.py).

`wav-16000-mono-190-frames.wav` is not made here: it is synthetic audio (two
sines and pink noise, made with FFmpeg) of exactly 48 384 samples, so the
front-end gives 190 frames and a second patch would end on the last frame. Its
reference vector came from the same Essentia pipeline. Keep its entry when
writing fixtures.json again.

    python scripts/make_fixtures.py [--ffmpeg ffmpeg] [--out fixtures/track-embedding-1]
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

MASTER_RATE = 48000

# name, seconds, tempo (BPM), chord roots (MIDI), seed, FFmpeg output arguments
FIXTURES = [
    ("mp3-44100-stereo.mp3", 8.0, 124, [57, 53, 48, 55], 1,
     ["-ac", "2", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "128k"],
     "MP3, 44.1 kHz, stereo"),
    ("aac-48000-stereo.m4a", 8.0, 118, [50, 46, 53, 48], 2,
     ["-ac", "2", "-ar", "48000", "-c:a", "aac", "-b:a", "128k"],
     "AAC in M4A, 48 kHz, stereo"),
    ("wav-16000-mono.wav", 8.0, 128, [52, 48, 55, 50], 3,
     ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"],
     "WAV 16-bit PCM, 16 kHz, mono (no resampling)"),
    ("flac-22050-mono.flac", 8.0, 122, [55, 52, 48, 50], 4,
     ["-ac", "1", "-ar", "22050", "-c:a", "flac"],
     "FLAC, 22.05 kHz, mono"),
    ("wav-44100-stereo-one-patch.wav", 2.1, 126, [53, 57], 5,
     ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le"],
     "WAV 16-bit PCM, 44.1 kHz, stereo, just above one patch (one patch)"),
]


def _midi_hz(note):
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def _one_pole_lowpass(x, cutoff, rate):
    a = np.exp(-2.0 * np.pi * cutoff / rate)
    out = np.empty_like(x)
    acc = 0.0
    for i, value in enumerate(x):  # short signals; clarity over speed
        acc = (1.0 - a) * value + a * acc
        out[i] = acc
    return out


def synthesise(seconds, bpm, roots, seed, rate=MASTER_RATE):
    """``(n, 2)`` float64 stereo in [-0.8, 0.8]."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * rate))
    left = np.zeros(n)
    right = np.zeros(n)
    beat = 60.0 / bpm

    def hit(start_s, length_s):
        i0 = int(start_s * rate)
        i1 = min(n, i0 + int(length_s * rate))
        return i0, i1, np.arange(i1 - i0) / rate

    n_beats = int(seconds / beat) + 1
    for b in range(n_beats):
        # kick on every beat: falling sine
        i0, i1, tt = hit(b * beat, 0.35)
        if i1 > i0:
            freq = 50.0 + 100.0 * np.exp(-tt * 30.0)
            phase = 2 * np.pi * np.cumsum(freq) / rate
            kick = np.sin(phase) * np.exp(-tt * 9.0) * 0.9
            left[i0:i1] += kick
            right[i0:i1] += kick
        # snare on beats 2 and 4: noise burst + tone, a little left
        if b % 2 == 1:
            i0, i1, tt = hit(b * beat, 0.25)
            if i1 > i0:
                noise = np.diff(rng.standard_normal(i1 - i0 + 1))
                snare = (0.35 * noise + 0.3 * np.sin(2 * np.pi * 185 * tt)) * np.exp(-tt * 18.0)
                left[i0:i1] += 0.6 * snare
                right[i0:i1] += 0.4 * snare
        # hi-hats on eighths: differenced noise, short, right
        for half in (0.0, 0.5):
            i0, i1, tt = hit((b + half) * beat, 0.06)
            if i1 > i0:
                noise = np.diff(np.diff(rng.standard_normal(i1 - i0 + 2)))
                hat = 0.12 * noise * np.exp(-tt * 60.0)
                left[i0:i1] += 0.3 * hat
                right[i0:i1] += 0.7 * hat

    bar = 4 * beat
    bass = np.zeros(n)
    pad_l = np.zeros(n)
    pad_r = np.zeros(n)
    for k in range(int(seconds / bar) + 1):
        root = roots[k % len(roots)]
        # bass: eighth notes on the root (octave below), saw from harmonics
        for e in range(8):
            i0, i1, tt = hit(k * bar + e * beat / 2, beat / 2 * 0.9)
            if i1 > i0:
                f0 = _midi_hz(root - 12 + (7 if e == 6 else 0))
                saw = sum(np.sin(2 * np.pi * h * f0 * tt) / h for h in range(1, 9))
                bass[i0:i1] += 0.25 * saw * np.exp(-tt * 6.0)
        # pad: minor-ish triad + seventh, slightly detuned per side
        i0, i1, tt = hit(k * bar, bar)
        if i1 > i0:
            env = np.minimum(1.0, tt / 0.15) * np.exp(-tt * 0.6)
            for interval, gain in ((0, 0.10), (3, 0.08), (7, 0.08), (10, 0.05)):
                f0 = _midi_hz(root + interval)
                for h, hg in ((1, 1.0), (2, 0.4), (3, 0.2)):
                    pad_l[i0:i1] += gain * hg * env * np.sin(2 * np.pi * h * f0 * 0.998 * tt)
                    pad_r[i0:i1] += gain * hg * env * np.sin(2 * np.pi * h * f0 * 1.002 * tt + 0.3)
    bass = _one_pole_lowpass(bass, 900.0, rate)
    left += bass + pad_l
    right += bass + pad_r
    stereo = np.stack([left, right], axis=1)
    return 0.8 * stereo / np.abs(stereo).max()


def encode(stereo, out_path, args, ffmpeg):
    command = [ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-y",
               "-f", "f32le", "-ar", str(MASTER_RATE), "-ac", "2", "-i", "-",
               "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact",
               *args, out_path]
    result = subprocess.run(command, input=stereo.astype("<f4").tobytes(), capture_output=True)
    if result.returncode != 0:
        raise SystemExit(f"FFmpeg failed for {out_path}: {result.stderr.decode(errors='replace')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "track-embedding-1"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    listing = []
    for name, seconds, bpm, roots, seed, ffmpeg_args, description in FIXTURES:
        path = os.path.join(args.out, name)
        encode(synthesise(seconds, bpm, roots, seed), path, ffmpeg_args, args.ffmpeg)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        listing.append({"audio": name, "audio_sha256": digest, "description": description,
                        "reference": os.path.splitext(name)[0] + ".npy"})
        print(f"{name}: {os.path.getsize(path)} bytes", file=sys.stderr)
    with open(os.path.join(args.out, "fixtures.json"), "w", encoding="utf-8") as handle:
        json.dump({"version": "track-embedding/1", "fixtures": listing}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
