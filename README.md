# track-embedding

The shared definition of a track embedding for Harmonia and fomoplayer: a
versioned specification ([SPECIFICATION.md](SPECIFICATION.md)), a Python
reference implementation both applications pin, and conformance fixtures.

Status: draft. Nothing here is implemented yet; the plan is below.

## Plan

1. `track_embedding` Python package (Python 3.11+, numpy, scipy, samplerate,
   onnxruntime; FFmpeg found by the caller): `decode(path)`, `embed(samples)`,
   `embed_file(path)`, `VERSION = "track-embedding/1"`, and a model downloader
   that verifies sha256 and requires the caller to pass the user's licence
   acceptance.
2. `constants/` with the mel front-end constants and the pinned model hashes.
3. `fixtures/track-embedding-1/` with synthetic and freely licensed audio and the
   reference vectors, plus a conformance test both applications run.
4. Both applications pin a tagged release of this repository
   (`track-embedding @ git+https://…@v1.0.0`).

The models are licensed by their authors (MTG, CC BY-NC-SA 4.0) and are never
stored in this repository.
