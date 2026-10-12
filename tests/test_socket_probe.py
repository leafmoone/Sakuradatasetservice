"""Mock regressions for cleanup decisions; these do not exercise OS sockets."""

import errno

import pytest

from sakuradatasetservice.service import DataServiceError, _prepare_socket_path


@pytest.mark.parametrize(
    "error",
    [
        PermissionError(errno.EACCES, "denied"),
        TimeoutError("busy"),
        BlockingIOError(errno.EAGAIN, "backlog full"),
    ],
)
def test_ambiguous_socket_failure_preserves_path(tmp_path, monkeypatch, error):
    path = tmp_path / "socket"
    path.write_bytes(b"existing endpoint")

    class Probe:
        closed = False

        def settimeout(self, value):
            pass

        def connect(self, target):
            raise error

        def close(self):
            self.closed = True

    probe = Probe()
    monkeypatch.setattr(
        "sakuradatasetservice.service.socket.socket", lambda *args: probe
    )
    monkeypatch.setattr("sakuradatasetservice.service.stat.S_ISSOCK", lambda mode: True)
    with pytest.raises(DataServiceError, match="safely checked"):
        _prepare_socket_path(path)
    assert path.read_bytes() == b"existing endpoint"
    assert probe.closed


@pytest.mark.parametrize("disappears", [False, True])
def test_confirmed_stale_or_missing_socket(tmp_path, monkeypatch, disappears):
    path = tmp_path / "socket"
    path.write_bytes(b"stale endpoint")

    class Probe:
        def settimeout(self, value):
            pass

        def connect(self, target):
            if disappears:
                path.unlink()
                raise FileNotFoundError(errno.ENOENT, "gone")
            raise ConnectionRefusedError(errno.ECONNREFUSED, "stale")

        def close(self):
            pass

    monkeypatch.setattr(
        "sakuradatasetservice.service.socket.socket", lambda *args: Probe()
    )
    monkeypatch.setattr("sakuradatasetservice.service.stat.S_ISSOCK", lambda mode: True)
    _prepare_socket_path(path)
    assert not path.exists()


def test_socket_replaced_during_probe_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / "socket"
    path.write_bytes(b"old endpoint")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"new endpoint")

    class Probe:
        def settimeout(self, value):
            pass

        def connect(self, target):
            replacement.replace(path)
            raise ConnectionRefusedError(errno.ECONNREFUSED, "old endpoint stale")

        def close(self):
            pass

    monkeypatch.setattr(
        "sakuradatasetservice.service.socket.socket", lambda *args: Probe()
    )
    monkeypatch.setattr("sakuradatasetservice.service.stat.S_ISSOCK", lambda mode: True)
    with pytest.raises(DataServiceError, match="changed"):
        _prepare_socket_path(path)
    assert path.read_bytes() == b"new endpoint"
