"""``constants/MANIFEST``: the sha256 of every file that changes the vector.

Each non-comment line is ``sha256 size file-name source``, where ``source`` is
``bundled`` (the file ships in ``constants/``) or the URL a model file is
downloaded from.
"""

import hashlib
import os
from dataclasses import dataclass
from importlib import resources

from .errors import ModelHashMismatchError, ModelsNotFoundError

CONSTANTS_DIR = os.fspath(resources.files(__package__) / "constants")
MANIFEST_PATH = os.path.join(CONSTANTS_DIR, "MANIFEST")

MEL_FRONTEND = "mel_frontend.npz"
EFFNET_MODEL = "discogs_multi_embeddings-effnet-bs64-1.onnx"

_HASH_BLOCK = 1 << 20


@dataclass(frozen=True)
class PinnedFile:
    name: str
    sha256: str
    size: int
    source: str  # "bundled" or a download URL

    @property
    def bundled(self):
        return self.source == "bundled"


def _read_manifest(path=MANIFEST_PATH):
    files = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path}:{number}: expected 4 columns, got {len(parts)}")
            sha256, size, name, source = parts
            files[name] = PinnedFile(name, sha256, int(size), source)
    return files


FILES = _read_manifest()


def model_files():
    """The pinned files that are downloaded (not bundled), in manifest order."""
    return [f for f in FILES.values() if not f.bundled]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_path(directory, name):
    """Path of the pinned file ``name`` in ``directory`` after checking its
    sha256. Raises :class:`ModelsNotFoundError` when it is missing and
    :class:`ModelHashMismatchError` when it is another file."""
    pinned = FILES[name]
    path = os.path.join(os.fspath(directory), name)
    if not os.path.isfile(path):
        hint = ("" if pinned.bundled else
                " Download it with track_embedding.download_models(target_dir, "
                "licence_accepted=True) after the user accepts the models' licence "
                "(CC BY-NC-SA 4.0).")
        raise ModelsNotFoundError(f"{name} is not in {os.path.abspath(path)!r}.{hint}")
    actual = sha256_of(path)
    if actual != pinned.sha256:
        raise ModelHashMismatchError(
            f"{os.path.abspath(path)!r} has sha256 {actual}, but track-embedding/1 "
            f"pins {pinned.sha256} ({pinned.size} bytes): it is another file.")
    return path
