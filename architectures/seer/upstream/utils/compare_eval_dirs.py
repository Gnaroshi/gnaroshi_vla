import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


_WINDOW_LINES = 30
_STRIDE = 3
_MAX_RATES = _WINDOW_LINES // _STRIDE

_PERCENT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")
_TASK_RE = re.compile(r"^Success rates for task (\d+) (.*):$")


def _extract_success_rate(log_path: Path) -> Tuple[float, List[float]]:
    lines = log_path.read_text().splitlines()
    parsed_rev: List[float] = []

    for line in reversed(lines):
        match = _PERCENT_RE.search(line)
        if not match:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if not (0.0 <= value <= 100.0):
            continue
        parsed_rev.append(value)
        if len(parsed_rev) >= _MAX_RATES:
            break

    parsed = list(reversed(parsed_rev))
    if not parsed:
        raise ValueError(f"No success rates parsed from {log_path}")
    return sum(parsed) / len(parsed), parsed


def _extract_task_names(log_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for line in log_path.read_text().splitlines():
        match = _TASK_RE.match(line.strip())
        if match:
            names[int(match.group(1))] = match.group(2)
    return names


def _load_dir(path: Path) -> Dict[str, Tuple[float, List[float]]]:
    rows: Dict[str, Tuple[float, List[float]]] = {}
    for child in sorted(path.iterdir()):
        if not child.is_file():
            continue
        try:
            rows[child.name] = _extract_success_rate(child)
        except Exception:
            continue
    if not rows:
        raise ValueError(f"No valid log files found in {path}")
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pstdev(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    mean = _mean(xs)
    return (sum((x - mean) ** 2 for x in xs) / len(xs)) ** 0.5


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def _sanitize_task_name(name: str) -> str:
    return name[:-1] if name.endswith(")") else name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, type=str, help="First eval directory")
    parser.add_argument("--b", required=True, type=str, help="Second eval directory")
    parser.add_argument("--label-a", default="A", type=str, help="Display label for --a")
    parser.add_argument("--label-b", default="B", type=str, help="Display label for --b")
    parser.add_argument("--topk", default=5, type=int, help="How many improved/degraded tasks to print")
    args = parser.parse_args()

    dir_a = Path(args.a)
    dir_b = Path(args.b)
    rows_a = _load_dir(dir_a)
    rows_b = _load_dir(dir_b)

    common = sorted(set(rows_a) & set(rows_b))
    only_a = sorted(set(rows_a) - set(rows_b))
    only_b = sorted(set(rows_b) - set(rows_a))
    if not common:
        raise ValueError("The two directories do not share any common log filenames")

    task_names = _extract_task_names(dir_b / common[0])
    srs_a = [rows_a[name][0] for name in common]
    srs_b = [rows_b[name][0] for name in common]

    print("Overall")
    print(
        f"{args.label_a}: mean={_fmt_pct(_mean(srs_a))}, "
        f"stdev={_fmt_pct(_pstdev(srs_a))}, min={_fmt_pct(min(srs_a))}, max={_fmt_pct(max(srs_a))}"
    )
    print(
        f"{args.label_b}: mean={_fmt_pct(_mean(srs_b))}, "
        f"stdev={_fmt_pct(_pstdev(srs_b))}, min={_fmt_pct(min(srs_b))}, max={_fmt_pct(max(srs_b))}"
    )
    print(f"Delta ({args.label_b} - {args.label_a}): {_fmt_pct(_mean(srs_b) - _mean(srs_a))}")

    if only_a:
        print(f"Only in {args.label_a}: {', '.join(only_a)}")
    if only_b:
        print(f"Only in {args.label_b}: {', '.join(only_b)}")

    print("\nCheckpoint Diffs")
    print(f"{'ckpt':<12} {'A':>8} {'B':>8} {'B-A':>8}")
    for name in common:
        sr_a = rows_a[name][0]
        sr_b = rows_b[name][0]
        diff = sr_b - sr_a
        print(f"{name:<12} {sr_a:>7.2f}% {sr_b:>7.2f}% {diff:>+7.2f}%")

    task_len = min(len(rows_a[common[0]][1]), len(rows_b[common[0]][1]))
    task_rows: List[Tuple[int, float, float, float, str]] = []
    for idx in range(task_len):
        avg_a = _mean([rows_a[name][1][idx] for name in common])
        avg_b = _mean([rows_b[name][1][idx] for name in common])
        diff = avg_b - avg_a
        task_name = _sanitize_task_name(task_names.get(idx, f"task_{idx}"))
        task_rows.append((idx, avg_a, avg_b, diff, task_name))

    print("\nTask Diffs")
    print(f"{'task':<6} {'A':>8} {'B':>8} {'B-A':>8}  name")
    for idx, avg_a, avg_b, diff, task_name in task_rows:
        print(f"{idx:<6} {avg_a:>7.2f}% {avg_b:>7.2f}% {diff:>+7.2f}%  {task_name}")

    sorted_rows = sorted(task_rows, key=lambda x: x[3], reverse=True)
    topk = max(1, min(args.topk, len(sorted_rows)))

    print(f"\nMost Improved Tasks ({args.label_b} - {args.label_a})")
    for idx, avg_a, avg_b, diff, task_name in sorted_rows[:topk]:
        print(f"{idx}: {task_name} | {avg_a:.2f}% -> {avg_b:.2f}% | {diff:+.2f}%")

    print(f"\nMost Degraded Tasks ({args.label_b} - {args.label_a})")
    for idx, avg_a, avg_b, diff, task_name in sorted_rows[-topk:]:
        print(f"{idx}: {task_name} | {avg_a:.2f}% -> {avg_b:.2f}% | {diff:+.2f}%")


if __name__ == "__main__":
    main()
