"""download_models: refuses without the licence acceptance, verifies sha256,
leaves no partial file."""

import contextlib
import functools
import http.server
import os
import threading
import urllib.request

import pytest

from track_embedding import DownloadError, LicenceNotAcceptedError, download_models, manifest


@contextlib.contextmanager
def _serve(directory):
    requests = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):  # keep a record instead of printing
            requests.append(self.path)

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("accepted", [False, None, 1, "yes", "True"])
def test_refuses_unless_the_licence_was_accepted(tmp_path, monkeypatch, accepted):
    def no_network(*args, **kwargs):
        raise AssertionError("no download may start without the licence acceptance")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    target = tmp_path / "models"
    with pytest.raises(LicenceNotAcceptedError, match="CC BY-NC-SA 4.0"):
        download_models(target, licence_accepted=accepted)
    assert not target.exists()


def test_wrong_bytes_are_refused_and_nothing_is_left(tmp_path):
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    for pinned in manifest.model_files():
        (mirror / pinned.name).write_bytes(b"x" * pinned.size)
    target = tmp_path / "models"
    with _serve(mirror) as (url, _):
        with pytest.raises(DownloadError, match="sha256"):
            download_models(target, licence_accepted=True, base_url=url)
    assert os.listdir(target) == []


def test_unreachable_server_is_a_download_error(tmp_path):
    with _serve(tmp_path) as (url, _):  # serves an empty folder: 404
        with pytest.raises(DownloadError, match="could not download"):
            download_models(tmp_path / "models", licence_accepted=True, base_url=url)
    assert os.listdir(tmp_path / "models") == []


def test_downloads_verified_files_and_keeps_good_ones(tmp_path, models_dir):
    target = tmp_path / "models"
    with _serve(models_dir) as (url, requests):
        download_models(target, licence_accepted=True, base_url=url)
        assert sorted(os.listdir(target)) == sorted(f.name for f in manifest.model_files())
        for pinned in manifest.model_files():
            assert manifest.sha256_of(target / pinned.name) == pinned.sha256
        fetched = len(requests)
        download_models(target, licence_accepted=True, base_url=url)
        assert len(requests) == fetched  # already present and verified: not fetched again
