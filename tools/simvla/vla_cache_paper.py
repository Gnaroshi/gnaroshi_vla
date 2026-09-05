"""Validate paired VLA-Cache paper rows and export four-suite table entries."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from architectures.simvla.adapters.vla_cache.eval import (
    _load_manifest, _read_jsonl, _sha256, _write_json, implementation_identity,
)

SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
TABLE_ORDER = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SEEDS = ("seed01", "seed02", "seed03")


def read(path):
    return json.loads(Path(path).read_text())


def references(args):
    long = read(args.long_reference)
    native = next(row for row in long["row_summaries"] if row["row"] == "full_nfe10")
    if native["seeds"] != 3 or native["episodes"] != 1500:
        raise ValueError("Long baseline requires 3 x 500 episodes")
    expected = {("libero_10", seed): {"manifest_sha256": long["manifest_sha256"][seed]} for seed in SEEDS}
    for path in (args.nonlong_seed1_reference, args.nonlong_seed23_reference):
        document = read(path)
        for cell in document["cell_reports"].values():
            if cell["row"] != "full_nfe10":
                continue
            if cell["episodes"] != 500 or cell["failures"] or cell["verdict"] != "PAPER_ROW_PASS":
                raise ValueError(f"incomplete native reference: {cell}")
            key = cell["suite"], cell["inference_seed"]
            if key in expected:
                raise ValueError(f"duplicate native reference: {key}")
            expected[key] = cell
    if set(expected) != {(suite, seed) for suite in SUITES for seed in SEEDS}:
        raise ValueError("references must cover all 12 suite/seed cells exactly once")
    return expected, native


def validate_manifests(args):
    expected, native = references(args)
    manifests = {}
    for suite in SUITES:
        for seed_index, seed in enumerate(SEEDS):
            if suite == "libero_10":
                base = Path(args.manifest_root)
            else:
                reference = args.nonlong_seed1_reference if seed == "seed01" else args.nonlong_seed23_reference
                base = Path(reference).parents[1] / "manifests"
            path = base / suite / seed / "episode_manifest.json"
            data = _load_manifest(path, row="vla_cache", max_episodes=None)
            keys = [(int(x["task_id"]), int(x["trial_id"])) for x in data["selected_episodes"]]
            if len(keys) != 500 or set(keys) != {(t, n) for t in range(10) for n in range(50)}:
                raise ValueError(f"manifest is not 10 tasks x 50 trials: {path}")
            if data["suite"] != suite or data["manifest_sha256"] != expected[suite, seed]["manifest_sha256"]:
                raise ValueError(f"manifest differs from existing baseline: {path}")
            if data["determinism_seed"] != 20260815 + seed_index:
                raise ValueError(f"unexpected determinism seed: {path}")
            manifests[suite, seed] = data
    return manifests, expected, native


def validate_cell(root, suite, seed, manifest):
    path = Path(root) / suite / "vla_cache" / seed
    item = read(path / "summary.json")
    metadata = read(path / "environment_metadata.json")
    if item.get("verdict") != "SIMVLA_VLA_CACHE_LIBERO_EVAL_COMPLETE" or item.get("episodes") != 500:
        raise ValueError(f"incomplete evaluation: {path}")
    if item.get("suite") != suite or item.get("implementation_identity") != implementation_identity():
        raise ValueError(f"source or suite mismatch: {path}")
    if item["manifest_file_sha256"] != manifest["manifest_file_sha256"]:
        raise ValueError(f"manifest file mismatch: {path}")
    if metadata["implementation_identity"] != item["implementation_identity"]:
        raise ValueError(f"metadata source mismatch: {path}")
    if metadata["hostname"] != "jbr-TRX50" or "5090" not in metadata["gpu"]:
        raise ValueError(f"not rb2 RTX5090: {path}")
    if metadata["renderer"]["MUJOCO_GL"] != "egl" or metadata["model_dtype"] != "torch.float32" or metadata["mujoco"] != "2.3.7":
        raise ValueError(f"renderer/precision/MuJoCo mismatch: {path}")
    rows = _read_jsonl(path / "progress.jsonl")
    specs = {(int(x["task_id"]), int(x["trial_id"])): x for x in manifest["selected_episodes"]}
    keys = [(x["task_id"], x["trial_id"]) for x in rows]
    if len(keys) != 500 or set(keys) != set(specs):
        raise ValueError(f"duplicate or missing episode outcomes: {path}")
    for row in rows:
        spec = specs[row["task_id"], row["trial_id"]]
        if row["suite"] != suite or any(row[key] != spec[key] for key in ("init_state_index", "environment_seed")):
            raise ValueError(f"episode identity mismatch: {path}")
        if row["num_policy_queries"] != (row["episode_length"] + 4) // 5:
            raise ValueError(f"R5 query count mismatch: {path}")
        if row["num_action_transformer_calls"] != row["num_policy_queries"] * 10:
            raise ValueError(f"10-step flow count mismatch: {path}")
    successes = sum(bool(x["success"]) for x in rows)
    if item["successes"] != successes or abs(item["success_rate"] - successes / 500) > 1e-12:
        raise ValueError(f"summary/outcomes disagree: {path}")
    latency = mean(x["policy_latency_mean_ms"] for x in rows)
    if abs(item["latency_per_executed_action_ms"] - latency) > 1e-8:
        raise ValueError(f"latency aggregation mismatch: {path}")
    return dict(suite=suite, seed=seed, episodes=500, successes=successes,
                success_rate_percent=100 * successes / 500, latency_ms_per_action=latency,
                summary_path=str(path / "summary.json"), summary_sha256=_sha256(path / "summary.json"))


def aggregate(rows, baseline):
    by_suite = {}
    for suite in SUITES:
        cells = [row for row in rows if row["suite"] == suite]
        if len(cells) != 3 or {row["seed"] for row in cells} != set(SEEDS):
            raise ValueError(f"three complete distinct seeds required: {suite}")
        rates = [row["success_rate_percent"] for row in cells]
        latencies = [row["latency_ms_per_action"] for row in cells]
        by_suite[suite] = dict(sr_mean=mean(rates), sr_sample_std=stdev(rates),
                               latency_mean=mean(latencies), latency_sample_std=stdev(latencies))
    average = mean(by_suite[suite]["sr_mean"] for suite in SUITES)
    return dict(suites=by_suite, four_suite_sr_mean=average,
                historical_long_speedup=baseline["seed_mean_latency_per_action_ms"] / by_suite["libero_10"]["latency_mean"])


def main(args):
    manifests, expected, native = validate_manifests(args)
    if args.command == "preflight":
        print("FOUR_SUITE_MANIFEST_PASS: 12 matched cells, 500 episodes each")
        return
    if args.command == "complete":
        validate_cell(args.output, args.suite, args.seed, manifests[args.suite, args.seed])
        return
    rows = [validate_cell(args.output, suite, seed, manifests[suite, seed]) for suite in SUITES for seed in SEEDS]
    result = aggregate(rows, native)
    baseline_avg = mean([100 * native["seed_mean_success_rate"]] +
                        [mean(100 * expected[suite, seed]["success_rate"] for seed in SEEDS) for suite in TABLE_ORDER[:-1]])
    result.update(verdict="SIMVLA_VLA_CACHE_FOUR_SUITE_COMPLETE", episodes=6000, per_seed=rows,
                  baseline_four_suite_sr_mean=baseline_avg, delta_average_pp=result["four_suite_sr_mean"] - baseline_avg,
                  implementation_identity=implementation_identity(),
                  spread_definition="sample standard deviation across 3 evaluation seeds (ddof=1)",
                  latency_scope="mean of episode mean policy latency; synchronized policy.act, excludes env.step/rendering; historical native denominator, not a new paired timing trial",
                  baseline_rerun=False, baseline_long_latency_ms=native["seed_mean_latency_per_action_ms"])
    output = Path(args.output) / "summary"
    _write_json(output / "paper_summary.json", result)
    with (output / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def pm(value, spread, digits=1):
        return f"${value:.{digits}f}{{\\scriptstyle\\pm{spread:.{digits}f}}}$"
    long = result["suites"]["libero_10"]
    performance = " & ".join(pm(result["suites"][s]["sr_mean"], result["suites"][s]["sr_sample_std"]) for s in TABLE_ORDER)
    table_a = f"SimVLA & VLA-Cache & {performance} & {result['four_suite_sr_mean']:.1f} & {result['delta_average_pp']:+.1f} \\\\\n"
    table_b = "SimVLA & VLA-Cache & -- & 10 & " + pm(long["sr_mean"], long["sr_sample_std"]) + " & " + pm(long["latency_mean"], long["latency_sample_std"], 2) + f" & ${result['historical_long_speedup']:.2f}\\times$ & 0 \\\\\n"
    (output / "main_table_rows.tex").write_text("% SimVLA adaptation; 3 seeds x 500 episodes per suite, RTX5090/EGL.\n% K_C is not a periodic full-backbone schedule for this token-caching method.\n% Panel (a)\n" + table_a + "% Panel (b)\n" + table_b)
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "complete", "summary"))
    p.add_argument("--output", required=True)
    p.add_argument("--manifest-root", required=True)
    p.add_argument("--long-reference", required=True)
    p.add_argument("--nonlong-seed1-reference", required=True)
    p.add_argument("--nonlong-seed23-reference", required=True)
    p.add_argument("--suite", choices=SUITES)
    p.add_argument("--seed", choices=SEEDS)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
