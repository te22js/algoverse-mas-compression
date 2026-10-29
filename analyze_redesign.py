#!/usr/bin/env python3
"""Offline analysis of the redesign's existing config/calls/runs streams.

No backend is constructed, no token vocabulary is loaded, and primary scores
are read, never replaced. PR #4's task-cluster uncertainty is reused through
analysis_stats; the redesign remains responsible for execution and scoring.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

from analysis_stats import cluster_interval, paired_interval
import baseline
from stage_diagnostics import diagnose

VERSION = "redesign-offline-analysis-v1"
PREFIX = "Handoff from Agent 1:\n"
PROMPTED = {"unconditional", "destination", "instruction_strip", "allowlist"}
PROTOCOL_FIELDS = (
    "model", "compressor_model", "temperature", "seed", "handoff_regime",
    "experiment_mode", "compressor_budget_tokens", "primary_cer_min",
    "prompt_hashes", "token_count_method", "git_commit", "suites",
    "attack_enabled", "controls_enabled", "synthetic_expansion", "cer_strata",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_config(value):
    """The upstream strong CER stratum has an infinite upper bound."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    return value


def _jsonl(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: invalid JSON; no rows discarded") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _key(row, cfg, ids):
    if row.get("run_id") != cfg["run_id"]:
        raise ValueError("row run_id disagrees with config")
    tid = row.get("task_id")
    if tid not in ids:
        raise ValueError(f"unexpected selected task: {tid}")
    rep = _integer(row.get("repeat"), "repeat")
    if rep >= cfg["repeats"]:
        raise ValueError("repeat outside configured range")
    return tid, rep


def _dataset_provenance(cfg, tasks_path):
    """Check a clean historical task blob when available; never invent a hash."""
    name = cfg.get("tasks_file")
    if not isinstance(name, str) or Path(name).name != tasks_path.name:
        raise ValueError("--tasks disagrees with config.tasks_file")
    commit = cfg.get("git_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        return "unverified: recorded commit is not a full object ID"
    reference = f"{commit}:{name}"
    result = subprocess.run(
        ["git", "show", reference], cwd=Path(__file__).resolve().parent,
        capture_output=True, check=False)
    if result.returncode:
        return "unverified: historical dataset blob unavailable"
    expected = hashlib.sha256(result.stdout).hexdigest()
    if expected != _digest(tasks_path):
        raise ValueError("analysis dataset differs from the recorded commit's task file")
    if cfg.get("git_dirty"):
        return "unverified: matches commit, but executed tree was dirty"
    return "matches clean recorded commit"


def _stage_text(calls, condition):
    by = {c["stage"]: c for c in calls}
    expected = {"agent1", "agent2"} | ({"compressor"} if condition in PROMPTED else set())
    warnings = ["missing_" + stage + "_trace" for stage in sorted(expected - set(by))]
    a1, a2 = by.get("agent1", {}), by.get("agent2", {})
    raw = a1.get("raw_output")
    handoff = None
    if isinstance(raw, str) and not a1.get("api_error"):
        try:
            parsed = baseline._parse_tolerant(raw)
            handoff = parsed.get("handoff_note", "") or raw
            if not isinstance(handoff, str):
                handoff = None
                warnings.append("agent1_handoff_not_text")
        except (TypeError, ValueError, AttributeError):
            warnings.append("agent1_output_unrecoverable")
    transmitted = None
    user = a2.get("user_input")
    if isinstance(user, str):
        if not user.startswith(PREFIX):
            raise ValueError("Agent 2 input has an unsupported boundary; cannot infer transmitted text")
        transmitted = user[len(PREFIX):]
    comp = by.get("compressor")
    if condition == "none":
        if comp:
            raise ValueError("condition=none unexpectedly contains a compressor call")
        if handoff is not None and transmitted is not None and handoff != transmitted:
            raise ValueError("uncompressed handoff differs from Agent 2 input")
    elif condition in PROMPTED:
        if not comp:
            pass  # already exposed in the expected-stage coverage above
        elif handoff is not None and comp.get("user_input") != handoff:
            raise ValueError("compressor input differs from reconstructed Agent 1 handoff")
        if comp and isinstance(comp.get("raw_output"), str) and transmitted is not None:
            if comp["raw_output"].strip() != transmitted:
                raise ValueError("compressor output differs from Agent 2 input")
    elif condition != "sliding_window":
        raise ValueError(f"unsupported condition: {condition}")
    elif comp:
        raise ValueError("mechanical sliding_window unexpectedly contains a model compressor call")
    elif handoff is not None and transmitted is not None:
        if baseline.compressors._sliding_window(handoff) != transmitted:
            raise ValueError("mechanical sliding_window output differs from Agent 2 input")
    return raw, handoff, transmitted, a2.get("raw_output"), warnings


def load_run(run_dir, tasks_path):
    """Load one run without guessing repeats from results.json array order."""
    root, tasks_path = Path(run_dir), Path(tasks_path)
    cfg = json.loads((root / "config.json").read_text())
    if not isinstance(cfg, dict) or not cfg.get("run_id") or not cfg.get("condition"):
        raise ValueError("config needs run_id and condition")
    if cfg["condition"] not in baseline.compressors.ALL_CONDITIONS:
        raise ValueError("unknown configured compression condition")
    for name in ("model", "compressor_model", "temperature", "handoff_regime", "experiment_mode",
                 "token_count_method", "git_commit", "prompt_hashes"):
        if name not in cfg or cfg[name] is None:
            raise ValueError(f"configuration missing comparison identity: {name}")
    if not isinstance(cfg["token_count_method"], str) or not cfg["token_count_method"]:
        raise ValueError("configuration must identify its token counting method")
    _integer(cfg.get("repeats"), "repeats", 1)
    _integer(cfg.get("compressor_budget_tokens"), "compressor_budget_tokens", 1)
    for name in ("temperature", "primary_cer_min"):
        value = cfg.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"nonfinite or invalid configuration value: {name}")
    selected = cfg.get("task_ids")
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("config.task_ids must explicitly list unique selected tasks")
    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list) or any(not isinstance(t, dict) or not t.get("id") for t in tasks):
        raise ValueError("--tasks must contain task objects with IDs")
    task_map = {t["id"]: t for t in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("duplicate task IDs in dataset")
    if set(selected) - set(task_map):
        raise ValueError("configured task IDs are absent from analysis dataset")
    dataset_status = _dataset_provenance(cfg, tasks_path)
    if not (root / "runs.jsonl").exists():
        raise ValueError("runs.jsonl is required; results.json cannot identify repeats safely")
    by_key = {}
    for row in _jsonl(root / "runs.jsonl"):
        key = _key(row, cfg, set(selected))
        if key in by_key:
            raise ValueError(f"duplicate task/repeat in runs.jsonl: {key}")
        for field in ("condition", "model", "temperature", "git_commit", "handoff_regime",
                      "experiment_mode", "compressor_budget_tokens", "token_count_method"):
            if field in row and field in cfg and row[field] != cfg[field]:
                raise ValueError(f"run row/config mismatch: {field}")
        if not row.get("api_failure"):
            if type(row.get("task_success")) is not bool:
                raise ValueError("non-error primary task_success must be boolean")
            hit = _integer(row.get("required_hit"), "required_hit")
            total = _integer(row.get("required_total"), "required_total")
            task = task_map[key[0]]
            expected_total = sum(c["tool"] in set(task.get("agent2_held_tools", []))
                                 for c in task.get("required_calls", []))
            if total != expected_total:
                raise ValueError("saved required_total differs from the team's task write denominator")
            if hit > total or row["task_success"] != (hit == total):
                raise ValueError("inconsistent saved primary scoring fields")
            rate = row.get("call_hit_rate")
            if total and rate is not None and not math.isclose(rate, hit / total):
                raise ValueError("inconsistent saved call_hit_rate")
        for name in ("source_context_tokens", "agent1_input_tokens", "precompression_handoff_tokens", "compressed_tokens"):
            if row.get(name) is not None:
                _integer(row[name], name)
        for name in ("cer", "realized_ratio", "fraction_removed"):
            if row.get(name) is not None and (type(row[name]) not in (int, float) or not math.isfinite(row[name])):
                raise ValueError(f"nonfinite or invalid metric: {name}")
        if "suite" in row and row["suite"] != task_map[key[0]].get("domain"):
            raise ValueError("run suite disagrees with task snapshot")
        by_key[key] = row
    calls = collections.defaultdict(list)
    stages = set()
    all_call_usage = {name: 0 for name in ("prompt_tokens", "output_tokens")}
    missing_usage = collections.Counter()
    for call in _jsonl(root / "calls.jsonl"):
        key = _key(call, cfg, set(selected))
        stage = call.get("stage")
        if stage not in {"agent1", "agent2", "compressor"}:
            raise ValueError("unknown model stage")
        if (key, stage) in stages:
            raise ValueError("ambiguous duplicate model stage for task/repeat")
        stages.add((key, stage))
        expected_model = cfg["compressor_model"] if stage == "compressor" else cfg["model"]
        if call.get("model") != expected_model or call.get("temperature") != cfg["temperature"]:
            raise ValueError("call model/temperature disagrees with configured stage")
        if call.get("requested") is not True or type(call.get("succeeded")) is not bool:
            raise ValueError("call transport flags must be explicit booleans")
        if call["succeeded"] != (call.get("api_error") is None):
            raise ValueError("inconsistent call transport success/api_error flags")
        method = call.get("token_count_method")
        if method and cfg.get("token_count_method") and method != cfg["token_count_method"]:
            raise ValueError("mixed token counting methods in calls/config")
        for name in all_call_usage:
            value = call.get(name)
            if value is None:
                missing_usage[name] += 1
            else:
                all_call_usage[name] += _integer(value, name)
        calls[key].append(call)
    rows = []
    for key, row in sorted(by_key.items()):
        task = task_map[key[0]]
        raw, handoff, transmitted, a2_raw, warnings = _stage_text(calls[key], cfg["condition"])
        info = diagnose(task, agent1_raw=raw, handoff=handoff,
                        compressed=transmitted, agent2_raw=a2_raw,
                        tool_calls=row.get("tool_calls"), condition=cfg["condition"],
                        api_failure=bool(row.get("api_failure")))
        if any(c.get("api_error") for c in calls[key]) and not row.get("api_failure"):
            raise ValueError("failed model call conflicts with successful pipeline record")
        usage = {name: sum(c.get(name) or 0 for c in calls[key])
                 for name in ("prompt_tokens", "output_tokens")}
        rows.append({"record": row, "diagnostics": info, "boundary_warnings": warnings,
                     "usage": usage, "handoff": handoff, "transmitted": transmitted})
    planned = {(tid, rep) for tid in selected for rep in range(cfg["repeats"])}
    return {"config": cfg, "rows": rows,
            "missing": sorted(planned - set(by_key)),
            "orphan_calls": sorted(set(calls) - set(by_key)),
            "dataset_status": dataset_status,
            "all_call_usage": all_call_usage,
            "missing_usage_counts": dict(missing_usage),
            "input_hashes": {name: _digest(root / name) for name in
                             ("config.json", "runs.jsonl", "calls.jsonl", "results.json")
                             if (root / name).exists()},
            "path": str(root.resolve()), "tasks": task_map}


def _values(rows, field):
    values = collections.defaultdict(list)
    for item in rows:
        row = item["record"]
        if not row.get("api_failure"):
            value = row.get(field)
            if value is not None:
                values[row["task_id"]].append(float(value))
    return dict(values)


def _summarise(run, draws, seed):
    rows, cfg = run["rows"], run["config"]
    scored = [i for i in rows if not i["record"].get("api_failure")]
    hits = sum(i["record"]["required_hit"] for i in scored)
    total = sum(i["record"]["required_total"] for i in scored)
    counters = collections.Counter()
    for item in rows:
        counters.update(item["diagnostics"].get("counts", {}))
    groups = {}
    for field in ("suite", "compression_stratum"):
        groups[field] = {}
        for value in sorted({i["record"].get(field, "unknown") for i in rows}):
            group = [i for i in rows if i["record"].get(field, "unknown") == value]
            groups[field][value] = cluster_interval(_values(group, "task_success"), draws=draws, seed=seed)
    repeat_rates = {}
    for rep in range(cfg["repeats"]):
        valid = [i["record"] for i in scored if i["record"]["repeat"] == rep]
        repeat_rates[str(rep)] = sum(r["task_success"] for r in valid) / len(valid) if valid else None
    compressed = [i["record"].get("compressed_tokens") for i in scored
                  if i["record"].get("compressed_tokens") is not None]
    cer = [i["record"].get("cer") for i in scored if i["record"].get("cer") is not None]
    return {"condition": cfg["condition"], "run_id": cfg["run_id"],
            "selected_tasks": len(cfg["task_ids"]),
            "expected_trials": len(cfg["task_ids"]) * cfg["repeats"],
            "observed_trials": len(rows), "scored_trials": len(scored),
            "missing_trials": len(run["missing"]),
            "api_or_pipeline_failures": len(rows) - len(scored),
            "orphan_call_groups": len(run["orphan_calls"]),
            "scored_trials_with_incomplete_stage_traces": sum(bool(i["boundary_warnings"]) for i in scored),
            "task_success": cluster_interval(_values(rows, "task_success"), draws=draws, seed=seed),
            "observed_task_success_rate": sum(i["record"]["task_success"] for i in scored) / len(scored) if scored else None,
            "macro_task_call_completion": cluster_interval(_values(rows, "call_hit_rate"), draws=draws, seed=seed),
            "required_call_hit_rate": hits / total if total else None,
            "required_hits": hits, "required_calls": total,
            "repeat_task_success": repeat_rates,
            "group_task_success": groups,
            "diagnostic_counts": dict(counters),
            "compressed_length_observations": len(compressed),
            "compressed_tokens_median": statistics.median(compressed) if compressed else None,
            "over_nominal_budget": sum(n > cfg["compressor_budget_tokens"] for n in compressed),
            "cer_median": statistics.median(cer) if cer else None,
            "local_text_token_counts": run["all_call_usage"],
            "missing_local_usage_counts": run["missing_usage_counts"],
            "token_count_method": cfg.get("token_count_method"),
            "usage_is_billed_provider_usage": False}


def analyze(run_dirs, tasks_path, *, draws=2000, seed=1729, fixture=False):
    if type(draws) is not int or draws < 1:
        raise ValueError("bootstrap draws must be positive")
    runs = [load_run(path, tasks_path) for path in run_dirs]
    seen, run_ids, protocol = set(), set(), None
    for run in runs:
        cfg = run["config"]
        if cfg["run_id"] in run_ids:
            raise ValueError("duplicate run_id across input directories")
        run_ids.add(cfg["run_id"])
        if cfg["condition"] in seen:
            raise ValueError("one run directory per condition is required for unambiguous repeat pairing")
        seen.add(cfg["condition"])
        signature = {key: cfg.get(key) for key in PROTOCOL_FIELDS}
        signature["task_ids"] = cfg["task_ids"]
        signature["tasks_file"] = cfg["tasks_file"]
        if protocol is not None and signature != protocol:
            differences = [k for k in signature if signature[k] != protocol[k]]
            raise ValueError("incompatible run configurations: " + ", ".join(differences))
        protocol = signature
    summaries = [_summarise(run, draws, seed) for run in runs]
    base = next((run for run in runs if run["config"]["condition"] == "none"), None)
    comparisons = []
    def slots(run):
        return {(i["record"]["task_id"], i["record"]["repeat"]): float(i["record"]["task_success"])
                for i in run["rows"] if not i["record"].get("api_failure")}
    if base:
        for run in runs:
            if run is base:
                continue
            comparison = paired_interval(slots(base), slots(run), draws=draws, seed=seed)
            base_slots, treatment_slots = slots(base), slots(run)
            common = set(base_slots) & set(treatment_slots)
            comparison["per_task"] = [
                {"task_id": tid, "n_pairs": len(keys),
                 "baseline_rate": statistics.mean(base_slots[k] for k in keys),
                 "treatment_rate": statistics.mean(treatment_slots[k] for k in keys),
                 "delta": statistics.mean(treatment_slots[k] - base_slots[k] for k in keys)}
                for tid in sorted({k[0] for k in common})
                for keys in [[k for k in sorted(common) if k[0] == tid]]]
            comparison.update(condition=run["config"]["condition"], baseline_condition="none",
                              interpretation="matched task/repeat outcomes; pairing does not establish shared upstream draws or a causal mediation estimate")
            comparisons.append(comparison)
    return {"analysis_version": VERSION,
            "data_kind": "deterministic_fixture_non_empirical" if fixture else "backend_unverified_from_upstream_metadata",
            "conditions": summaries, "comparisons": comparisons,
            "baseline_available": base is not None,
            "provenance": {"analysis_tasks_path": str(Path(tasks_path).resolve()),
                           "analysis_tasks_sha256": _digest(tasks_path),
                           "analysis_dataset_task_count": len(runs[0]["tasks"]) if runs else 0,
                           "analysis_code_sha256": {name: _digest(Path(__file__).resolve().parent / name)
                                                    for name in ("analyze_redesign.py", "analysis_stats.py", "stage_diagnostics.py",
                                                                 "baseline.py", "failure_stage.py", "obtainable.py", "pipeline.py",
                                                                 "compressors.py", "suite_tools.py", "config.py")},
                           "python_version": sys.version.split()[0],
                           "bootstrap_draws": draws, "bootstrap_seed": seed,
                           "interval": "95% percentile bootstrap; whole tasks resampled with their repeats",
                           "config_serialization": "nonfinite upstream CER bounds are represented as strings; input hashes preserve exact original bytes",
                           "runs": [{k: _json_config(run[k]) for k in ("path", "config", "input_hashes", "dataset_status", "missing", "orphan_calls")} for run in runs]},
            "limitations": [
                "Scores are saved team primary scores, never rescored by this analysis.",
                "All selected tasks remain in coverage; API/pipeline failures and missing rows are not model failures.",
                "Stage diagnostics are observable value-presence candidates, not proof of semantic sufficiency or causation.",
                "PLAN schema warnings and configuration coverage warnings never remove primary rows.",
                "calls.parse_status is not a parser outcome; use runs.parse_events and raw evidence.",
                "Token counts are local text lengths, not billed usage; nominal budget is not a hard token limit.",
                "Upstream config does not record whether --mock was used; non-fixture backend identity remains unverified.",
                "Task-cluster intervals are exploratory; repeated calls are not independent tasks, and small samples can yield degenerate intervals.",
            ]}, runs


def write_outputs(analysis, runs, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("analysis output directory must be empty")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    columns = ["run_id", "task_id", "repeat", "condition", "suite", "task_success",
               "required_hit", "required_total", "call_hit_rate", "api_failure",
               "parse_events", "precompression_handoff_tokens", "compressed_tokens",
               "compressor_budget_tokens", "cer", "compression_stratum", "token_count_method"]
    with (out / "trials.csv").open("w", newline="") as handle, (out / "diagnostics.jsonl").open("w") as detail:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for item in run["rows"]:
                row = item["record"]
                writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in columns})
                detail.write(json.dumps({"run_id": row["run_id"], "task_id": row["task_id"],
                                         "repeat": row["repeat"], "condition": row["condition"],
                                         "boundary_warnings": item["boundary_warnings"],
                                         **item["diagnostics"]}, allow_nan=False) + "\n")
    def percent(value):
        return "unavailable" if value is None else f"{100 * value:.1f}%"
    lines = ["# Compression redesign offline analysis", "", analysis["data_kind"], "",
             "The team primary scores are unchanged. Rates below use available scored observations; coverage is reported separately.", "",
             "| Condition | Scored / planned | Missing | API or pipeline errors | Observed success | Equal-task success and 95% interval | Required-call completion |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["conditions"]:
        ci = row["task_success"]
        interval = f"{percent(ci['estimate'])} [{percent(ci['low'])}, {percent(ci['high'])}]"
        lines.append(f"| {row['condition']} | {row['scored_trials']} / {row['expected_trials']} | {row['missing_trials']} | {row['api_or_pipeline_failures']} | {percent(row['observed_task_success_rate'])} | {interval} | {percent(row['required_call_hit_rate'])} |")
    lines += ["", "Observed success weights available trials equally. Equal-task success averages repeats within each task before averaging tasks; these can differ when coverage is incomplete.", "",
              "## Matched comparisons", ""]
    if not analysis["baseline_available"]:
        lines += ["No no-compression baseline was supplied; no baseline differences are available.", ""]
    for comparison in analysis["comparisons"]:
        lines += [f"- {comparison['condition']} minus none: {percent(comparison['estimate'])} (absolute percentage-point difference); 95% interval [{percent(comparison['low'])}, {percent(comparison['high'])}], {comparison['n_pairs']} matched slots across {comparison['n_tasks']} tasks."]
    lines += ["", "Per-task paired differences and coverage are in analysis.json. Pairs match task/repeat IDs, not shared Agent 1 samples. No primary cohort is selected by observed success or schema warnings.", "", "## Interpretation limits", ""]
    lines += ["- " + text for text in analysis["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--tasks", required=True, help="explicit benchmark snapshot; never default to a legacy task set")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fixture", action="store_true", help="label deterministic or mock inputs as non-empirical")
    args = parser.parse_args(argv)
    try:
        analysis, runs = analyze(args.run_dirs, args.tasks, draws=args.bootstrap_draws,
                                 seed=args.seed, fixture=args.fixture)
        write_outputs(analysis, runs, args.out_dir)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"analysis rejected: {error}", file=sys.stderr)
        return 2
    print(f"Wrote validated offline analysis to {args.out_dir}; no model calls made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
