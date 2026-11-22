# track-embedding

The shared definition of a track embedding for Harmonia and fomoplayer: a
versioned specification ([SPECIFICATION.md](SPECIFICATION.md)), a Python
reference implementation both applications pin, and conformance fixtures.

Version 1 (`track-embedding/1`) reproduces the vectors fomoplayer has stored
since 2025 (Essentia `MonoLoader` + `TensorflowPredictEffnetDiscogs`) without
Essentia or TensorFlow: FFmpeg, libsamplerate, numpy and ONNX Runtime.

Status: version 1 is released; use `v1.0.1` or later (v1.0.0 counted one patch too
many for one audio length in 62, see the specification's "Corrections").

## Install

From a release tag (Python 3.11 or newer; FFmpeg is installed separately):

```sh
pip install "track-embedding @ git+https://github.com/gadgetmies/track-embedding@v1.0.1"
```

Dependencies: numpy, scipy, samplerate (libsamplerate), onnxruntime.

## Use

```python
import track_embedding

# Only after the user has accepted the models' licence (CC BY-NC-SA 4.0,
# non-commercial; see track_embedding.LICENCE_SUMMARY and LICENCE_URL):
track_embedding.download_models("models", licence_accepted=True)

samples = track_embedding.decode("track.mp3")          # mono float32 at 16 kHz
vector = track_embedding.embed(samples, "models")      # float32, 1280 values
vector = track_embedding.embed_file("track.mp3", "models", ffmpeg="/opt/homebrew/bin/ffmpeg")
track_embedding.VERSION                                # "track-embedding/1"
```

- `decode(path, ffmpeg="ffmpeg")` streams FFmpeg's output: memory holds the mono
  16 kHz result and one chunk, never the whole multi-channel decode.
- `embed(samples, models_dir)` loads the model once per process (use
  `Embedder(models_dir, threads=None)` for explicit control) and checks its
  sha256 against `track_embedding/constants/MANIFEST`.
- `download_models(target_dir, licence_accepted)` refuses unless
  `licence_accepted is True`, verifies size and sha256, and leaves no partial
  file. `base_url=` points it at a mirror holding the same files.
- Errors: `DecodeError` (carries FFmpeg's own message), `AudioTooShortError`
  (shorter than one patch, about 2 s), `ModelsNotFoundError`,
  `ModelHashMismatchError`, `LicenceNotAcceptedError` and `DownloadError`, all
  subclasses of `TrackEmbeddingError`.

## Develop and test

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test]'
# Once, after accepting the models' licence (CC BY-NC-SA 4.0):
.venv/bin/python -c "import track_embedding; track_embedding.download_models('models', licence_accepted=True)"
TRACK_EMBEDDING_MODELS_DIR=models .venv/bin/python -m pytest
```

`models/` is git-ignored; the model files are never committed. Tests that run
the model read `TRACK_EMBEDDING_MODELS_DIR` and **fail** (they do not skip) when
it is unset, so a run without the models cannot look green. The conformance test
alone:

```sh
TRACK_EMBEDDING_MODELS_DIR=models .venv/bin/python -m pytest tests/test_conformance.py -v
```

## Fixtures

`fixtures/track-embedding-1/` holds five short synthetic, music-like files
(chords, noise-burst drums, bass) in the formats the specification lists, made by
`scripts/make_fixtures.py`, and their reference vectors (`*.npy`), made with
`scripts/make_reference_vectors.py` by fomoplayer's Essentia pipeline, the
authority for version 1. `reference.json` records the fomoplayer commit, the
Essentia version, the model graph's sha256 and the FFmpeg version that made them.
An implementation conforms when its vector for every fixture has cosine
similarity of at least 0.99999 with the reference.

## Comparison with fomoplayer on real files

`scripts/compare_with_essentia.py` embeds a list of audio files with fomoplayer's
pipeline and with this package and reports the cosine similarity per file. The
run of 2026-11-22 on 50 DJ-library files is in
[results/2026-11-22-essentia-comparison.md](results/2026-11-22-essentia-comparison.md).

## Licence

The specification, code and fixtures are MIT-licensed ([LICENSE](LICENSE)). The
models the package downloads are not covered by it: they keep their authors'
licence (CC BY-NC-SA 4.0).
