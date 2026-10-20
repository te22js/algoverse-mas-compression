"""
Rescore saved runs with the current scorer. No API calls.

Every run saves the exact tool calls the agents made, so when the scorer
changes there's no need to pay for the run again:

    python3 rescore.py                      # results.json (or its .partial.jsonl)
    python3 rescore.py run1_4omini.json     # any saved results file

Prints the same table as run.py, a per-repeat line, and which tasks failed.
"""

from __future__ import annotations
import json
import os
import sys
from collections import defaultdict

from environment import ToolLog, score_run


def load(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        out = []
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out
    with open(path) as f:
        return json.load(f)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.json"
    if not os.path.exists(path) and os.path.exists(path + ".partial.jsonl"):
        path += ".partial.jsonl"
    tasks_path = sys.argv[2] if len(sys.argv) > 2 else "tasks/sample_tasks.json"

    records = load(path)
    with open(tasks_path) as f:
        tasks = {t["id"]: t for t in json.load(f)}

    # Keep only the newest prompt version, so old broken runs don't leak in.
    versions = [r.get("prompt_version", 0) for r in records]
    latest = max(versions) if versions else 0
    records = [r for r in records if r.get("prompt_version", 0) == latest]

    agg = defaultdict(lambda: {"n": 0, "task": 0, "attack": 0})
    by_rep = defaultdict(lambda: [0, 0])
    fails = defaultdict(int)
    for r in records:
        log = ToolLog(calls=r.get("calls", []))
        s = score_run(log, tasks[r["task_id"]])
        a = agg[r["condition"]]
        a["n"] += 1
        a["task"] += int(s["task_success"])
        a["attack"] += int(s["attack_fired"])
        k = (r["condition"], r.get("repeat", 0))
        by_rep[k][0] += int(s["task_success"])
        by_rep[k][1] += 1
        if not s["task_success"]:
            fails[(r["condition"], r["task_id"])] += 1

    model = records[0].get("model", "?") if records else "?"
    print(f"rescored {len(records)} runs from {path} "
          f"(model {model}, prompt v{latest})\n")
    print(f"{'condition':<18} {'attack rate':>12} {'task rate':>11}   (n)")
    print("-" * 48)
    for cond, a in agg.items():
        n = a["n"] or 1
        print(f"{cond:<18} {a['attack']/n:>12.0%} {a['task']/n:>11.0%}   ({a['n']})")

    print("\nper repeat (task rate):")
    for cond in agg:
        reps = sorted(r for c, r in by_rep if c == cond)
        print(f"  {cond:<18} " + "  ".join(
            f"{by_rep[(cond, r)][0]}/{by_rep[(cond, r)][1]}" for r in reps))

    if fails:
        print("\nfailed (times failed across repeats):")
        for (cond, tid), n in sorted(fails.items(), key=lambda x: -x[1]):
            print(f"  {cond:<18} {tid:<8} {n}")


if __name__ == "__main__":
    main()
