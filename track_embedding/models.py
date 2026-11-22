"""Download the version 1 model files, verified by sha256.

The models are the Music Technology Group's (Universitat Pompeu Fabra),
licensed CC BY-NC-SA 4.0 (non-commercial). The package never ships them: an
application downloads them after its user accepts that licence and passes the
acceptance here.
"""

import hashlib
import os
import tempfile
import urllib.error
import urllib.request

from . import manifest
from .errors import DownloadError, LicenceNotAcceptedError

LICENCE = "CC BY-NC-SA 4.0"
LICENCE_URL = "https://essentia.upf.edu/models.html"
LICENCE_SUMMARY = (
    "Essentia models by the Music Technology Group, Universitat Pompeu Fabra, "
    "licensed CC BY-NC-SA 4.0 for non-commercial use.")

_CHUNK = 1 << 16
_TIMEOUT_S = 60


def download_models(target_dir, licence_accepted, *, base_url=None, progress=None):
    """Download every model file pinned in ``constants/MANIFEST`` into
    ``target_dir`` and return ``target_dir``.

    ``licence_accepted`` must be ``True`` (the user accepted CC BY-NC-SA 4.0);
    anything else raises :class:`LicenceNotAcceptedError` before any network
    access. A file already present with the pinned sha256 is kept. Each file is
    written to a temporary file beside its destination and renamed into place
    only after its size and sha256 match, so a failure (raised as
    :class:`DownloadError`) leaves no partial file.

    ``base_url`` replaces the folder part of each pinned URL (a mirror holding
    the same files); ``progress(file_name, bytes_done, bytes_total)`` is called
    as bytes arrive."""
    if licence_accepted is not True:
        raise LicenceNotAcceptedError(
            f"the track-embedding/1 models are licensed {LICENCE} ({LICENCE_URL}); "
            f"pass licence_accepted=True only after the user has accepted that licence")
    target_dir = os.fspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    for pinned in manifest.model_files():
        destination = os.path.join(target_dir, pinned.name)
        if os.path.isfile(destination) and manifest.sha256_of(destination) == pinned.sha256:
            continue
        url = pinned.source if base_url is None else base_url.rstrip("/") + "/" + pinned.name
        _download_one(url, destination, pinned, progress)
    return target_dir


def _download_one(url, destination, pinned, progress):
    handle, temporary = tempfile.mkstemp(prefix=f".{pinned.name}.", suffix=".part",
                                         dir=os.path.dirname(destination))
    try:
        digest, done = hashlib.sha256(), 0
        with os.fdopen(handle, "wb") as out:
            try:
                with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response:
                    for block in iter(lambda: response.read(_CHUNK), b""):
                        out.write(block)
                        digest.update(block)
                        done += len(block)
                        if done > pinned.size:
                            raise DownloadError(
                                f"{url} sent more than the pinned {pinned.size} bytes")
                        if progress:
                            progress(pinned.name, done, pinned.size)
            except (urllib.error.URLError, OSError) as error:
                raise DownloadError(f"could not download {url}: {error}") from error
        if done != pinned.size:
            raise DownloadError(f"{url} sent {done} bytes; {pinned.size} are pinned")
        if digest.hexdigest() != pinned.sha256:
            raise DownloadError(f"{url} has sha256 {digest.hexdigest()}; "
                                f"{pinned.sha256} is pinned")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.remove(temporary)
