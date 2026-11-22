"""Make the reference vectors of fixtures/track-embedding-1/ with fomoplayer's
Essentia pipeline, the authority for track-embedding/1.

That pipeline is fomoplayer's ``analyser/embedding.py``: FFmpeg decodes the file
to a 16-bit PCM WAV at its own rate and layout (``audio_decode.to_wav``),
Essentia's ``MonoLoader(sampleRate=16000, resampleQuality=4)`` loads it, and
``TensorflowPredictEffnetDiscogs`` (output ``PartitionedCall:1``) gives the
per-patch rows, averaged. Every fixture goes through ``to_wav`` (fomoplayer's
own ``embed_file`` reads a ``.wav`` directly, which is the same for 16-bit PCM).

Run it with the Python of fomoplayer's analyser (Essentia + TensorFlow
installed) from this repository's root:

    /path/to/fomoplayer/analyser/venv/bin/python scripts/make_reference_vectors.py \\
        --analyser /path/to/fomoplayer/analyser

It writes ``<fixture>.npy`` (float32, 1280) beside each audio file and
``reference.json`` recording what made them.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "fixtures", "track-embedding-1")


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _git_head(path):
    result = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"cannot read the fomoplayer commit: {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analyser", required=True, help="fomoplayer's analyser folder")
    parser.add_argument("--fixtures", default=FIXTURES)
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.analyser))
    import essentia
    import embedding  # fomoplayer's analyser/embedding.py
    from audio_decode import to_wav

    model = embedding.build_model()
    graph = os.path.join(embedding.MODELS_DIR, f"{embedding.DEFAULT_MODEL}.pb")
    with open(os.path.join(args.fixtures, "fixtures.json"), encoding="utf-8") as handle:
        listing = json.load(handle)["fixtures"]
    with tempfile.TemporaryDirectory() as work:
        for fixture in listing:
            audio = os.path.join(args.fixtures, fixture["audio"])
            if _sha256(audio) != fixture["audio_sha256"]:
                raise SystemExit(f"{audio} does not match fixtures.json")
            wav = to_wav(audio, os.path.join(work, "decoded.wav"))
            vector = embedding.embed_wav(wav, model)
            if vector is None:
                raise SystemExit(f"{audio} is too short for one patch")
            np.save(os.path.join(args.fixtures, fixture["reference"]),
                    np.asarray(vector, dtype=np.float32))
            print(f"{fixture['audio']}: {len(vector)} values", file=sys.stderr)
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    record = {
        "made_by": "fomoplayer analyser/embedding.py (Essentia MonoLoader + "
                   "TensorflowPredictEffnetDiscogs, output PartitionedCall:1, mean over patches)",
        "fomoplayer_commit": _git_head(args.analyser),
        "essentia_version": essentia.__version__,
        "model_graph": os.path.basename(graph),
        "model_graph_sha256": _sha256(graph),
        "ffmpeg": ffmpeg.splitlines()[0] if ffmpeg else "unknown",
    }
    with open(os.path.join(args.fixtures, "reference.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
