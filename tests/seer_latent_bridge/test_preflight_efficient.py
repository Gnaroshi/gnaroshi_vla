import os

from tools.seer_latent_bridge.preflight_efficient import _available_memory_bytes


def test_available_memory_uses_linux_memavailable(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemFree:             16 kB\nMemAvailable:       32768 kB\n",
        encoding="utf-8",
    )

    def unexpected_sysconf(_name):
        raise AssertionError("sysconf fallback must not run when MemAvailable is valid")

    monkeypatch.setattr(os, "sysconf", unexpected_sysconf)
    assert _available_memory_bytes(meminfo) == 32768 * 1024


def test_available_memory_falls_back_to_sysconf(tmp_path, monkeypatch):
    missing_meminfo = tmp_path / "missing-meminfo"
    values = {"SC_PAGE_SIZE": 4096, "SC_AVPHYS_PAGES": 17}
    monkeypatch.setattr(os, "sysconf", values.__getitem__)
    assert _available_memory_bytes(missing_meminfo) == 4096 * 17
