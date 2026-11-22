# Track embedding specification

Status: **draft of version 1**, not implemented yet.

A track embedding is one vector that describes how a piece of audio sounds. Two
applications compute them, Harmonia (a desktop DJ library) and fomoplayer (a web
music discovery service), and each one must be able to search the other's
vectors. This document fixes every step that changes the vector, so that two
implementations of the same version produce the same vector, within the
tolerance below, from the same audio file.

A version never changes after it is released. Anything that changes the vector —
another model, another decoder, other front-end constants, another pooling —
is a new version with a new identifier. Implementations keep the identifier of
every stored vector next to it.

## Version 1: `track-embedding/1`

This version reproduces the vectors fomoplayer has stored since 2025 (Essentia's
`MonoLoader` + `TensorflowPredictEffnetDiscogs`) without Essentia or TensorFlow.

### 1. Decode

1. Decode the **first audio stream** of the file with FFmpeg to signed 16-bit
   little-endian PCM (`pcm_s16le`) at the stream's **own sample rate** and in its
   **own channel layout**. No FFmpeg resampling or channel conversion.
2. Convert to float32 by dividing by 32768.
3. Mix to mono as the arithmetic mean of the channels, `(L + R) / 2` for stereo.
   Never FFmpeg's `-ac 1`, which sums the channels.
4. Resample to 16 000 Hz with libsamplerate's **linear** converter
   (`SRC_LINEAR`), ratio `16000 / source_rate`. Skip this step when the source is
   already 16 000 Hz. Streaming the resampler block by block is allowed when it
   gives the same samples as one call over the whole signal.

### 2. Front-end

The log-mel front-end of Essentia's `TensorflowInputMusiCNN`:

- frames of 512 samples, hop 256; frame count `ceil(n / 256) + 1`; the signal is
  padded with 256 zeros in front and 512 zeros behind before framing;
- the window and the 96 × 257 mel matrix are the constants in
  `constants/mel_frontend.npz` (sha256 pinned in `constants/MANIFEST`);
- per frame: power spectrum of the windowed frame, mel bands, `log10(1 + 10000 · x)`.

### 3. Patches

Patches of 128 frames with a hop of 62 frames. Only whole patches: audio with
fewer than 128 frames has no embedding (implementations may set a higher
minimum length; that is an application rule, not part of the vector).

### 4. Model

`discogs_multi_embeddings-effnet-bs64-1` from the Music Technology Group (MTG) of
Universitat Pompeu Fabra, in its ONNX form, identified by sha256 (pinned in
`constants/MANIFEST`). The output is the 1280-wide `embeddings` layer per patch.
The published graph takes batches of exactly 64 patches; a shorter last batch is
zero-padded and the padding rows are discarded.

The model is licensed CC BY-NC-SA 4.0 (non-commercial). It is **downloaded by the
application after the user accepts that licence**, never shipped in a package.

### 5. Pooling

The embedding is the arithmetic mean of the per-patch vectors, float32, 1280
values, **not normalised**. Similarity is cosine, which ignores the length.

### 6. What audio is embedded

The vector describes the audio it was given. Version 1 does not fix which part of
a recording that is; each stored vector carries a `coverage` of `full` (the whole
file) or `preview` (a store preview). Whether a full-track vector and a preview
vector of the same recording are close enough to search across is measured, not
assumed (see "Open questions").

## Conformance

An implementation conforms to a version when, for every fixture in
`fixtures/<version>/`, the cosine similarity between its vector and the stored
reference vector is at least **0.99999**. Fixtures are synthetic or freely
licensed audio only, covering: 44.1 kHz stereo MP3, 48 kHz stereo AAC (M4A),
16 kHz mono WAV (no resampling), 22.05 kHz mono FLAC, and a file just above one
patch long. The reference vectors are produced once by the reference
implementation in this repository and are checked by both applications' test
suites.

Bit-identical vectors are not required: two libsamplerate builds or BLAS libraries
may differ in the last bits.

## Exchanging vectors

A vector sent between applications is `{ "version": "track-embedding/1",
"coverage": "full", "vector": [1280 numbers] }`. A receiver that does not hold
vectors of that version answers that it cannot search it, and lists the versions
it can; the sender then computes a vector of one of those versions if it can.

## Open questions for version 1

- Whether full-track and preview vectors of the same recording rank each other
  well enough, or whether Harmonia should embed a preview-like window for
  cross-application search. fomoplayer's research change
  `preview-vs-full-track-embeddings` measures exactly this.
