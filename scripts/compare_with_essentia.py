"""Compare the package with fomoplayer's Essentia pipeline on real audio files
(the local part of task 1.4 of the Harmonia change shared-track-embedding).

Two steps, each in its own Python environment, writing into one work folder:

1. With fomoplayer's analyser Python (Essentia + TensorFlow):

       /path/to/fomoplayer/analyser/venv/bin/python scripts/compare_with_essentia.py essentia \\
           --files files.txt --work scratch/compare --analyser /path/to/fomoplayer/analyser

   decodes each file exactly as fomoplayer does (``audio_decode.to_wav`` to a
   temporary WAV in the work folder, ``MonoLoader(sampleRate=16000,
   resampleQuality=4)``, ``TensorflowPredictEffnetDiscogs``, mean over patches)
   and saves the vector and a sha256 of the 16 kHz samples.

2. With this package's Python:

       .venv/bin/python scripts/compare_with_essentia.py package \\
           --files files.txt --work scratch/compare --models models --report result.md

   embeds each file with the package and writes a Markdown report: per file the
   format, sample rate, channels, duration, cosine with fomoplayer's vector and
   whether the decoded samples are bit-identical; then the minimum and median
   cosine. The report never names the files (they may be private).

``files.txt`` holds one audio path per line. The input files are only read.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def _paths(list_file):
    with open(list_file, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _samples_sha256(samples):
    return hashlib.sha256(np.ascontiguousarray(samples, dtype="<f4").tobytes()).hexdigest()


def run_essentia(args):
    sys.path.insert(0, os.path.abspath(args.analyser))
    from essentia.standard import MonoLoader
    import embedding  # fomoplayer's analyser/embedding.py
    from audio_decode import to_wav

    model = embedding.build_model()
    os.makedirs(args.work, exist_ok=True)
    for index, path in enumerate(_paths(args.files)):
        vector_path = os.path.join(args.work, f"{index:03d}.essentia.npy")
        if os.path.exists(vector_path):
            continue
        with tempfile.TemporaryDirectory(dir=args.work) as temporary:
            wav = to_wav(path, os.path.join(temporary, "decoded.wav"))
            samples = MonoLoader(filename=wav, sampleRate=16000, resampleQuality=4)()
            rows = model(samples)
        if len(rows) == 0:
            raise SystemExit(f"file {index} is shorter than one patch")
        np.save(vector_path, np.asarray(rows.T.mean(1), dtype=np.float32))
        with open(os.path.join(args.work, f"{index:03d}.essentia.json"), "w") as handle:
            json.dump({"samples": len(samples), "samples_sha256": _samples_sha256(samples)}, handle)
        print(f"essentia {index}", file=sys.stderr)


def _probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", path],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"ffprobe failed: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return (stream["codec_name"], int(stream["sample_rate"]), int(stream["channels"]),
            float(info["format"]["duration"]))


def run_package(args):
    import track_embedding

    embedder = track_embedding.Embedder(args.models)
    rows = []
    for index, path in enumerate(_paths(args.files)):
        with open(os.path.join(args.work, f"{index:03d}.essentia.json")) as handle:
            essentia = json.load(handle)
        reference = np.load(os.path.join(args.work, f"{index:03d}.essentia.npy"))
        samples = track_embedding.decode(path)
        vector = embedder(samples)
        cosine = float(np.dot(vector, reference) /
                       (np.linalg.norm(vector) * np.linalg.norm(reference)))
        identical = (len(samples) == essentia["samples"]
                     and _samples_sha256(samples) == essentia["samples_sha256"])
        codec, rate, channels, duration = _probe(path)
        rows.append((index, codec, rate, channels, duration, cosine, identical))
        print(f"package {index}: {cosine:.9f}", file=sys.stderr)
    cosines = np.array([row[5] for row in rows])
    lines = [
        "# Package vs fomoplayer's Essentia pipeline on real files",
        "",
        f"{len(rows)} files from a DJ library (names withheld). fomoplayer: "
        "`audio_decode.to_wav` + Essentia `MonoLoader(sampleRate=16000, resampleQuality=4)` + "
        "`TensorflowPredictEffnetDiscogs` (TensorFlow graph), mean over patches. Package: "
        f"`track_embedding` {track_embedding.VERSION} (ONNX Runtime). Whole files.",
        "",
        f"- minimum cosine: **{cosines.min():.9f}**",
        f"- median cosine: **{np.median(cosines):.9f}**",
        f"- files at or above the conformance tolerance 0.99999: "
        f"{int((cosines >= 0.99999).sum())} of {len(rows)}",
        f"- decoded 16 kHz samples bit-identical to Essentia's `MonoLoader`: "
        f"{sum(row[6] for row in rows)} of {len(rows)}",
        "",
        "| # | codec | sample rate | channels | duration (s) | cosine | samples identical |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, codec, rate, channels, duration, cosine, identical in rows:
        lines.append(f"| {index} | {codec} | {rate} | {channels} | {duration:.0f} | "
                     f"{cosine:.9f} | {'yes' if identical else 'no'} |")
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"min {cosines.min():.9f} median {np.median(cosines):.9f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="step", required=True)
    essentia = sub.add_parser("essentia")
    essentia.add_argument("--analyser", required=True)
    package = sub.add_parser("package")
    package.add_argument("--models", required=True)
    package.add_argument("--report", required=True)
    for step in (essentia, package):
        step.add_argument("--files", required=True)
        step.add_argument("--work", required=True)
    args = parser.parse_args()
    (run_essentia if args.step == "essentia" else run_package)(args)


if __name__ == "__main__":
    main()
