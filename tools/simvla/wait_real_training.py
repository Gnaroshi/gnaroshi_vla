"""Wait without CUDA initialization; never signal another experiment process."""

import argparse
import csv
import io
from pathlib import Path
import subprocess
import time


def process_token(pid, proc_root=Path("/proc")):
    try:
        # comm may contain spaces or parentheses. Fields after its final ')' start at state.
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[0] in {"Z", "X"}:
        return None
    return fields[19]  # starttime distinguishes an original process from a reused PID.


def gpu_is_idle(gpu_id):
    def query(option):
        return subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}", option, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout

    try:
        processes = query("--query-compute-apps=pid").strip()
        if processes:
            return False, "gpu_compute_pids=" + processes.replace("\n", ",")
        rows = list(csv.reader(io.StringIO(query("--query-gpu=memory.free,utilization.gpu"))))
        if len(rows) != 1 or len(rows[0]) != 2:
            raise ValueError("unexpected GPU status response")
        free_mib, utilization = map(int, rows[0])
        # rb2 has a 32 GiB RTX5090. This is a launch check, not a memory reservation.
        idle = free_mib >= 28000 and utilization <= 10
        return idle, f"gpu_free_mib={free_mib} gpu_utilization={utilization}"
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return False, f"gpu_status_unavailable={type(error).__name__}"


def wait_for_slot(pids, manifest, gpu_id=0, poll_seconds=30, idle_samples=2):
    tracked = {pid: process_token(pid) for pid in pids}
    print(f"WAIT_START parent_processes={tracked} dataset={manifest}", flush=True)
    print("Parent exit is not proof of experiment success; Doll is an independent task.", flush=True)
    consecutive_idle = 0
    previous_message = None
    last_print = 0.0
    while True:
        alive = [pid for pid, token in tracked.items() if token is not None and process_token(pid) == token]
        if alive:
            idle, message = False, f"waiting_for_pipeline_pids={alive}"
        elif not manifest.is_file():
            idle, message = False, "waiting_for_corrected_dataset_transfer"
        else:
            idle, message = gpu_is_idle(gpu_id)
        consecutive_idle = consecutive_idle + 1 if idle else 0
        if consecutive_idle >= idle_samples:
            print(f"TRAINING_SLOT_READY {message}; dataset preflight follows", flush=True)
            return
        now = time.monotonic()
        if message != previous_message or now - last_print >= 300:
            print(f"WAIT {message} idle_samples={consecutive_idle}/{idle_samples} retry_seconds={poll_seconds}", flush=True)
            previous_message, last_print = message, now
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pids", required=True)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    try:
        pids = [int(value) for value in args.parent_pids.split(",")]
        if not pids or any(pid <= 1 for pid in pids):
            raise ValueError
    except ValueError:
        parser.error("parent-pids must be comma-separated process IDs greater than 1")
    try:
        wait_for_slot(pids, args.dataset_manifest, args.gpu_id)
    except KeyboardInterrupt:
        print("WAIT_CANCELLED; no training started", flush=True)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
