# Package vs fomoplayer's Essentia pipeline on real files

50 files from a DJ library (names withheld). fomoplayer: `audio_decode.to_wav` + Essentia `MonoLoader(sampleRate=16000, resampleQuality=4)` + `TensorflowPredictEffnetDiscogs` (TensorFlow graph), mean over patches. Package: `track_embedding` track-embedding/1 (ONNX Runtime). Whole files.

- minimum cosine: **0.999999762**
- median cosine: **1.000000000**
- files at or above the conformance tolerance 0.99999: 50 of 50
- decoded 16 kHz samples bit-identical to Essentia's `MonoLoader`: 50 of 50

Notes:

- The library holds only 44.1 kHz and 48 kHz audio, so files 46-49 (22.05 kHz mono
  FLAC, 32 kHz stereo FLAC, 96 kHz stereo FLAC, 16 kHz mono WAV) were transcoded
  with FFmpeg, into a scratch folder, from library MP3s to cover the other
  resampling ratios. The other 46 are library files as they are (MP3, AAC in M4A
  at 44.1 and 48 kHz, ALAC, FLAC, Ogg Vorbis, 24-bit and 32-bit float WAV).
- "Samples identical" compares a sha256 of the 16 kHz float32 samples. They are
  identical because the package passes libsamplerate the ratio in float32, as
  Essentia does; the remaining difference (cosine above 0.9999997) is TensorFlow
  vs ONNX Runtime arithmetic in the model.
- Run with FFmpeg 9.0.2, Essentia 2.1-beta6-dev (fomoplayer commit c90c4298),
  onnxruntime 1.32.0, samplerate 0.2.4, on an Apple M2 Pro.

| # | codec | sample rate | channels | duration (s) | cosine | samples identical |
|---|---|---|---|---|---|---|
| 0 | mp3 | 44100 | 2 | 74 | 1.000000119 | yes |
| 1 | mp3 | 44100 | 2 | 81 | 1.000000119 | yes |
| 2 | mp3 | 44100 | 2 | 72 | 1.000000000 | yes |
| 3 | mp3 | 44100 | 2 | 71 | 1.000000000 | yes |
| 4 | mp3 | 44100 | 2 | 72 | 0.999999940 | yes |
| 5 | mp3 | 44100 | 2 | 493 | 0.999999940 | yes |
| 6 | mp3 | 44100 | 2 | 75 | 1.000000000 | yes |
| 7 | mp3 | 44100 | 2 | 73 | 0.999999881 | yes |
| 8 | mp3 | 44100 | 2 | 72 | 1.000000000 | yes |
| 9 | mp3 | 44100 | 2 | 244 | 1.000000119 | yes |
| 10 | mp3 | 44100 | 2 | 73 | 1.000000119 | yes |
| 11 | mp3 | 44100 | 2 | 72 | 1.000000119 | yes |
| 12 | mp3 | 44100 | 2 | 74 | 1.000000000 | yes |
| 13 | mp3 | 44100 | 2 | 510 | 0.999999881 | yes |
| 14 | mp3 | 44100 | 2 | 309 | 0.999999940 | yes |
| 15 | mp3 | 44100 | 2 | 72 | 1.000000000 | yes |
| 16 | mp3 | 44100 | 2 | 73 | 0.999999762 | yes |
| 17 | aac | 44100 | 2 | 391 | 0.999999940 | yes |
| 18 | aac | 44100 | 2 | 288 | 0.999999940 | yes |
| 19 | aac | 44100 | 2 | 290 | 1.000000000 | yes |
| 20 | aac | 44100 | 2 | 536 | 1.000000000 | yes |
| 21 | aac | 44100 | 2 | 315 | 1.000000000 | yes |
| 22 | aac | 44100 | 2 | 342 | 1.000000000 | yes |
| 23 | aac | 48000 | 2 | 374 | 1.000000000 | yes |
| 24 | aac | 48000 | 2 | 291 | 0.999999881 | yes |
| 25 | aac | 48000 | 2 | 349 | 1.000000119 | yes |
| 26 | aac | 48000 | 2 | 212 | 1.000000000 | yes |
| 27 | aac | 48000 | 2 | 315 | 1.000000000 | yes |
| 28 | aac | 48000 | 2 | 248 | 1.000000000 | yes |
| 29 | aac | 48000 | 2 | 353 | 1.000000000 | yes |
| 30 | aac | 48000 | 2 | 291 | 0.999999881 | yes |
| 31 | aac | 48000 | 2 | 574 | 0.999999881 | yes |
| 32 | aac | 48000 | 2 | 796 | 1.000000000 | yes |
| 33 | aac | 48000 | 2 | 102 | 1.000000119 | yes |
| 34 | alac | 44100 | 2 | 203 | 0.999999940 | yes |
| 35 | alac | 44100 | 2 | 371 | 1.000000119 | yes |
| 36 | alac | 44100 | 2 | 279 | 1.000000119 | yes |
| 37 | flac | 44100 | 2 | 413 | 1.000000000 | yes |
| 38 | flac | 44100 | 2 | 494 | 1.000000000 | yes |
| 39 | flac | 44100 | 2 | 201 | 0.999999940 | yes |
| 40 | pcm_f32le | 44100 | 2 | 96 | 0.999999881 | yes |
| 41 | pcm_f32le | 44100 | 2 | 74 | 1.000000000 | yes |
| 42 | pcm_s24le | 44100 | 2 | 60 | 0.999999881 | yes |
| 43 | pcm_s24le | 44100 | 2 | 60 | 1.000000119 | yes |
| 44 | vorbis | 44100 | 2 | 284 | 0.999999881 | yes |
| 45 | vorbis | 44100 | 2 | 519 | 0.999999940 | yes |
| 46 | flac | 22050 | 1 | 72 | 1.000000000 | yes |
| 47 | flac | 32000 | 2 | 72 | 0.999999940 | yes |
| 48 | flac | 96000 | 2 | 74 | 1.000000000 | yes |
| 49 | pcm_s16le | 16000 | 1 | 391 | 1.000000000 | yes |

## Speed: package vs Harmonia's current analysis

One 6-minute library MP3 (363 s, 44.1 kHz stereo, 192 kbit/s), median of three
runs after a warm-up, both models capped at 2 ONNX Runtime threads, Apple M2 Pro.
Harmonia: `audio_hash.decode_samples` (FFmpeg resamples to 16 kHz itself) +
`analysis_models.EffnetEmbedding` (onnxruntime 1.30.0). Package: `decode` +
`Embedder(threads=2)` (onnxruntime 1.32.0). Two alternating rounds:

| | decode | embed | total |
|---|---|---|---|
| Harmonia, round 1 | 0.56 s | 2.81 s | 3.37 s |
| package, round 1 | 0.88 s | 2.56 s | 3.44 s |
| Harmonia, round 2 | 0.59 s | 2.81 s | 3.40 s |
| package, round 2 | 0.72 s | 2.50 s | 3.23 s |

The package's decode is 0.15-0.3 s slower (FFmpeg writes 16-bit PCM at 44.1 kHz,
1.4 times the bytes of 16 kHz float, and libsamplerate resamples in Python's process); the model
step is the same work. In total the two are within run-to-run noise.
