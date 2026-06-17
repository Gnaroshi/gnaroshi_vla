import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


_PERCENT_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*%")
_OVERALL_SUCCESS_RE = re.compile(
    r"(?:success[_ ]rate|total[_ ]success|avg[_ ]success|average[_ ]success).*?"
    r"([-+]?\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)


@dataclass
class EvalRow:
    name: str
    sr: float
    task_rates: List[float]
    source: str


def _as_percent(value) -> float:
    v = float(value)
    if 0.0 <= v <= 1.0:
        return v * 100.0
    return v


def _looks_like_eval_json(path: str) -> bool:
    name = os.path.basename(path)
    return name.endswith("_eval.json") or name.endswith("eval.json")


def _parse_eval_json(path: str, display_name: str) -> EvalRow:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "success_rate" not in data:
        raise ValueError("missing top-level success_rate")

    task_rates: List[float] = []
    task_results = data.get("task_results", [])
    if isinstance(task_results, list):
        for item in task_results:
            if isinstance(item, dict) and "success_rate" in item:
                task_rates.append(_as_percent(item["success_rate"]))

    sr = _as_percent(data["success_rate"])
    if not task_rates:
        task_rates = _collect_json_nested_rates(data)
    return EvalRow(display_name, sr, task_rates, "json")


def _collect_json_nested_rates(data) -> List[float]:
    """Fallback for older summary JSONs that may not use task_results."""
    out: List[float] = []
    for key in ("category_results", "task_summary", "tasks"):
        values = data.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            for rate_key in ("success_rate", "avg_success", "success"):
                if rate_key in item and isinstance(item[rate_key], (int, float)):
                    out.append(_as_percent(item[rate_key]))
                    break
        if out:
            break
    return out


def _parse_task_rates_from_log(lines: List[str]) -> List[float]:
    task_rates: List[float] = []
    for idx, line in enumerate(lines):
        if "Success rates for task" not in line:
            continue
        for next_line in lines[idx + 1 : idx + 5]:
            m = _PERCENT_RE.search(next_line)
            if not m:
                continue
            task_rates.append(float(m.group(1)))
            break
    return task_rates


def _parse_overall_success_from_log(lines: List[str]) -> Optional[float]:
    matches: List[float] = []
    for line in lines:
        if "Success rates for task" in line:
            continue
        m = _OVERALL_SUCCESS_RE.search(line)
        if not m:
            continue
        value = float(m.group(1))
        if 0.0 <= value <= 100.0:
            matches.append(value)
    return matches[-1] if matches else None


def _parse_legacy_percent_window(lines: List[str], max_rates: int = 10) -> List[float]:
    parsed_rev: List[float] = []
    for line in reversed(lines):
        m = _PERCENT_RE.search(line)
        if not m:
            continue
        value = float(m.group(1))
        if 0.0 <= value <= 100.0:
            parsed_rev.append(value)
        if len(parsed_rev) >= max_rates:
            break
    return list(reversed(parsed_rev))


def _parse_log(path: str, display_name: str) -> EvalRow:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    task_rates = _parse_task_rates_from_log(lines)
    overall = _parse_overall_success_from_log(lines)

    if task_rates and overall is None:
        overall = sum(task_rates) / len(task_rates)

    if overall is not None:
        return EvalRow(display_name, overall, task_rates, "log")

    legacy_rates = _parse_legacy_percent_window(lines)
    if legacy_rates:
        return EvalRow(display_name, sum(legacy_rates) / len(legacy_rates), legacy_rates, "legacy-log")

    raise ValueError("no success rates parsed")


def _walk_files(root: str, recursive: bool) -> Iterable[str]:
    if not recursive:
        for fname in sorted(os.listdir(root)):
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                yield path
        return

    skip_dirs = {".git", "__pycache__", "eval_videos", "wandb"}
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in sorted(files):
            yield os.path.join(cur, fname)


def _display_name(path: str, root: Optional[str]) -> str:
    if root is not None:
        return os.path.relpath(path, root)
    return os.path.basename(path)


def _collect_rows(path: str, source: str, recursive: bool) -> List[EvalRow]:
    if os.path.isfile(path):
        name = _display_name(path, None)
        if source in ("auto", "json") and path.endswith(".json"):
            try:
                return [_parse_eval_json(path, name)]
            except Exception:
                if source == "json":
                    raise
        if source in ("auto", "log"):
            return [_parse_log(path, name)]
        return []

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    rows: List[EvalRow] = []
    warn_paths: List[Tuple[str, Exception]] = []

    if source in ("auto", "json"):
        for fpath in _walk_files(path, recursive):
            if not fpath.endswith(".json") or not _looks_like_eval_json(fpath):
                continue
            try:
                rows.append(_parse_eval_json(fpath, _display_name(fpath, path)))
            except Exception as e:
                warn_paths.append((fpath, e))

    if rows or source == "json":
        for fpath, e in warn_paths:
            print(f"[WARN] skipping {fpath}: {e}")
        return sorted(rows, key=lambda x: x.name)

    for fpath in _walk_files(path, recursive):
        if not fpath.endswith((".log", ".out", ".txt")):
            continue
        try:
            rows.append(_parse_log(fpath, _display_name(fpath, path)))
        except Exception as e:
            print(f"[WARN] skipping {fpath}: {e}")

    return sorted(rows, key=lambda x: x.name)


def fmt_tsr_list(xs: List[float], elem_w: int = 6) -> str:
    return "[" + ", ".join(f"{x:{elem_w}.2f}" for x in xs) + "]"


def per_task_average(tsr_lists: List[List[float]]) -> List[float]:
    valid = [x for x in tsr_lists if x]
    if not valid:
        return []
    length = min(len(x) for x in valid)
    return [sum(x[j] for x in valid) / len(valid) for j in range(length)]


def per_task_topk(rows: List[EvalRow], k: int) -> List[List[Tuple[str, float]]]:
    valid = [row for row in rows if row.task_rates]
    if not valid:
        return []
    length = min(len(row.task_rates) for row in valid)
    out: List[List[Tuple[str, float]]] = []
    for j in range(length):
        scores = [(row.name, row.task_rates[j]) for row in valid]
        scores.sort(key=lambda x: x[1], reverse=True)
        out.append(scores[:k])
    return out


def _print_rows(rows: List[EvalRow], topk: int):
    mean_sr_all = sum(row.sr for row in rows) / len(rows)
    min_sr_all = min(row.sr for row in rows)
    max_sr_all = max(row.sr for row in rows)

    elem_w = 6
    label_avg = "per task avg"
    label_minmax = "sr min/max"

    name_w = max(max(len(row.name) for row in rows), len(label_avg), len(label_minmax))
    src_w = max(len(row.source) for row in rows + [EvalRow("", 0.0, [], "source")])
    sr_minmax_s = f"{min_sr_all:.2f}%~{max_sr_all:.2f}%"
    sr_strs = [f"{row.sr:.2f}%" for row in rows] + [f"{mean_sr_all:.2f}%", sr_minmax_s]
    sr_w = max(len(s) for s in sr_strs)

    tsr_strs = [fmt_tsr_list(row.task_rates, elem_w=elem_w) for row in rows]
    tsr_w = max(len(s) for s in tsr_strs + ["tasks"])

    print(f"{'name':<{name_w}} | {'src':<{src_w}} | {'sr':>{sr_w}} | {'tasks':<{tsr_w}}")
    print("-" * name_w + "-+-" + "-" * src_w + "-+-" + "-" * sr_w + "-+-" + "-" * tsr_w)
    for row, tsr_s in zip(rows, tsr_strs):
        print(f"{row.name:<{name_w}} | {row.source:<{src_w}} | {row.sr:>{sr_w}.2f}% | {tsr_s:<{tsr_w}}")

    avg_list = per_task_average([row.task_rates for row in rows])
    avg_tsr_s = fmt_tsr_list(avg_list, elem_w=elem_w)
    tsr_w2 = max(tsr_w, len(avg_tsr_s))

    print(f"{label_avg:<{name_w}} | {'summary':<{src_w}} | {mean_sr_all:>{sr_w}.2f}% | {avg_tsr_s:<{tsr_w2}}")
    print(f"{label_minmax:<{name_w}} | {'summary':<{src_w}} | {sr_minmax_s:>{sr_w}} | {'':<{tsr_w2}}")

    effective_topk = min(topk, len(rows))
    pt_topk = per_task_topk(rows, effective_topk)
    if pt_topk:
        task_label_w = max(len(f"task {len(pt_topk) - 1}"), len(f"top{effective_topk} avg"))
        topk_lines = [
            ", ".join(f"{fname}({score:.2f})" for fname, score in tops)
            for tops in pt_topk
        ]
        topk_avg_list = [sum(score for _, score in tops) / len(tops) for tops in pt_topk]
        topk_avg_s = fmt_tsr_list(topk_avg_list, elem_w=elem_w)
        value_w = max(max((len(s) for s in topk_lines), default=0), len(topk_avg_s))

        print(f"\nPer-task Top {effective_topk}:")
        for j, s in enumerate(topk_lines):
            print(f"{('task ' + str(j)):<{task_label_w}} | {s:<{value_w}}")
        print("-" * task_label_w + "-+-" + "-" * value_w)
        print(f"{('top' + str(effective_topk) + ' avg'):<{task_label_w}} | {topk_avg_s:<{value_w}}")

    top = sorted(rows, key=lambda x: x.sr, reverse=True)[:effective_topk]
    top_sorted_by_name = sorted(top, key=lambda x: x.name)

    print(f"\nTop {effective_topk} by success rate (name order):")
    for row in top_sorted_by_name:
        print(f"{row.name:<{name_w}} | {row.source:<{src_w}} | {row.sr:>{sr_w}.2f}%")

    avg_top = sum(row.sr for row in top_sorted_by_name) / len(top_sorted_by_name)
    print(f"Average of top {len(top_sorted_by_name)}: {avg_top:.2f}%")


def main(args):
    rows = _collect_rows(args.dir, args.source, not args.no_recursive)
    if not rows:
        print(f"No valid eval files found in {args.dir}")
        return
    _print_rows(rows, args.topk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--source", choices=("auto", "json", "log"), default="auto")
    ap.add_argument("--no-recursive", action="store_true")
    args = ap.parse_args()
    main(args)
