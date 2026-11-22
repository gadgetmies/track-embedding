"""Reference implementation of the shared track embedding, version 1.

See SPECIFICATION.md in this repository. Quick use::

    import track_embedding
    track_embedding.download_models("models", licence_accepted=True)  # after the user accepts
    vector = track_embedding.embed_file("track.mp3", "models")         # float32, 1280 values

FFmpeg must be installed (pass its path as ``ffmpeg=`` when it is not on PATH).
"""

from .decoding import SAMPLE_RATE, decode
from .models import LICENCE, LICENCE_SUMMARY, LICENCE_URL, download_models
from .embedding import DIMENSIONS, Embedder, embed, embed_file
from .errors import (AudioTooShortError, DecodeError, DownloadError,
                     LicenceNotAcceptedError, ModelHashMismatchError,
                     ModelsNotFoundError, TrackEmbeddingError)

VERSION = "track-embedding/1"

__all__ = [
    "VERSION", "SAMPLE_RATE", "DIMENSIONS", "decode", "embed", "embed_file", "Embedder",
    "download_models", "LICENCE", "LICENCE_SUMMARY", "LICENCE_URL",
    "TrackEmbeddingError", "DecodeError", "AudioTooShortError", "ModelsNotFoundError",
    "ModelHashMismatchError", "LicenceNotAcceptedError", "DownloadError",
]
