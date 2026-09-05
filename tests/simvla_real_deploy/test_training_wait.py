from unittest.mock import Mock
import subprocess

from tools.simvla import wait_real_training as wait


def test_process_identity_handles_spaces_pid_reuse_and_zombies(tmp_path):
    proc = tmp_path / "123"
    proc.mkdir()
    stat = proc / "stat"
    for state, token, expected in [("S", "42", "42"), ("S", "43", "43"), ("Z", "43", None)]:
        stat.write_text("123 (shell (child)) " + " ".join([state] + ["0"] * 18 + [token]))
        assert wait.process_token(123, tmp_path) == expected
    assert wait.process_token(456, tmp_path) is None


def test_gpu_errors_never_look_idle(monkeypatch):
    monkeypatch.setattr(wait.subprocess, "run", Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 15)))
    assert wait.gpu_is_idle(0)[0] is False


def test_other_compute_process_blocks(monkeypatch):
    run = Mock(return_value=Mock(stdout="777\n"))
    monkeypatch.setattr(wait.subprocess, "run", run)
    assert wait.gpu_is_idle(0)[0] is False
    assert run.call_count == 1


def test_gpu_idle_thresholds(monkeypatch):
    for status, expected in [("30000, 0\n", True), ("27000, 0\n", False), ("30000, 50\n", False), ("N/A, 0\n", False)]:
        monkeypatch.setattr(wait.subprocess, "run", Mock(side_effect=[Mock(stdout=""), Mock(stdout=status)]))
        assert wait.gpu_is_idle(0)[0] is expected


def test_pipeline_dataset_and_stable_idle_are_all_required(monkeypatch, tmp_path):
    # Snapshot, one active-parent poll, then exit. No GPU query during the phase gap.
    token = Mock(side_effect=["initial", "initial", None, None, None, None, None])
    monkeypatch.setattr(wait, "process_token", token)
    manifest = tmp_path / "manifest.json"
    sleep_count = 0

    def sleep(_):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            manifest.write_text("{}")

    monkeypatch.setattr(wait.time, "sleep", sleep)
    gpu = Mock(side_effect=[(True, "idle"), (False, "busy"), (True, "idle"), (True, "idle")])
    monkeypatch.setattr(wait, "gpu_is_idle", gpu)
    wait.wait_for_slot([123], manifest, poll_seconds=0)
    assert sleep_count == 5
    assert gpu.call_count == 4


def test_reused_pid_does_not_block(monkeypatch, tmp_path):
    monkeypatch.setattr(wait, "process_token", Mock(side_effect=["old", "new", "new"]))
    monkeypatch.setattr(wait.time, "sleep", Mock())
    monkeypatch.setattr(wait, "gpu_is_idle", Mock(return_value=(True, "idle")))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    wait.wait_for_slot([123], manifest, poll_seconds=0)
