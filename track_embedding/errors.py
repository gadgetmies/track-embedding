"""The errors the package raises. Every failure is one of these (or an
``OSError`` from the file system); nothing falls back silently."""


class TrackEmbeddingError(Exception):
    """Base class of every error this package raises on purpose."""


class DecodeError(TrackEmbeddingError):
    """FFmpeg could not be started or could not decode the file. The message
    carries FFmpeg's own error output."""


class AudioTooShortError(TrackEmbeddingError, ValueError):
    """The audio is shorter than one model patch (128 front-end frames, a little
    over 2 s at 16 kHz), so it has no embedding."""


class ModelsNotFoundError(TrackEmbeddingError, FileNotFoundError):
    """The model file is not in the given folder. Download it with
    :func:`track_embedding.download_models` after the user accepts its licence."""


class ModelHashMismatchError(TrackEmbeddingError):
    """A model or constants file does not have the sha256 pinned in
    ``constants/MANIFEST``: it is another file than the one version 1 uses."""


class LicenceNotAcceptedError(TrackEmbeddingError, PermissionError):
    """:func:`track_embedding.download_models` was called without the user's
    acceptance of the models' licence (CC BY-NC-SA 4.0)."""


class DownloadError(TrackEmbeddingError):
    """A model download failed (network error, wrong size or wrong sha256).
    No partial file is left behind."""
