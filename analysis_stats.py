"""Task-cluster uncertainty for saved results; no model or scorer dependencies.

Adapted from PR #4's ``scripts/analyze_run2.py`` percentile/task-cluster
bootstrap mathematics (b6c5eb6). Every task has equal weight after averaging
its available repeats. Resampling preserves those repeats as one cluster.
Intervals describe empirical task variation, not independent repeat trials;
identical task means may yield a degenerate interval. Fewer than two tasks
cannot support a task-resampling interval, so their bounds are unavailable.
"""
from __future__ import annotations

from collections import defaultdict
import math
from numbers import Real
import random
from statistics import mean


def _validate_draws(draws):
    if type(draws) is not int or draws <= 0:
        raise ValueError("draws must be a positive integer")


def _finite(value):
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("observations must be finite numbers")
    return float(value)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_interval(values_by_task: dict[str, list[float]], *, draws=2000,
                     seed=1729) -> dict:
    """Mean and percentile 95% interval, with whole tasks resampled.

    Empty repeat lists contribute neither a task nor an observation. Input
    order does not affect the deterministic seeded result.
    """
    _validate_draws(draws)
    clusters = {task: [_finite(value) for value in values]
                for task, values in values_by_task.items()}
    means = [mean(clusters[task]) for task in sorted(clusters) if clusters[task]]
    count = len(means)
    result = {"estimate": mean(means) if means else None, "low": None,
              "high": None, "n_tasks": count,
              "n_observations": sum(len(values) for values in clusters.values())}
    if count >= 2:
        rng = random.Random(seed)
        sampled = [sum(means[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(draws)]
        result.update(low=_percentile(sampled, .025),
                      high=_percentile(sampled, .975))
    return result


def paired_interval(baseline: dict[tuple[str, int], float],
                    treatment: dict[tuple[str, int], float], *, draws=2000,
                    seed=1729) -> dict:
    """Treatment-minus-baseline differences on matched task/repeat slots.

    Match availability is reported rather than filling missing slots. Tasks
    have equal weight even when their matched repeat counts differ. Pairing
    alone does not imply the arms shared an Agent 1 draw or random seed.
    Improvements/regressions count matched slots, not independent tasks.
    """
    _validate_draws(draws)
    before = {key: _finite(value) for key, value in baseline.items()}
    after = {key: _finite(value) for key, value in treatment.items()}
    common = before.keys() & after.keys()
    differences = defaultdict(list)
    regressions = improvements = 0
    for task, repeat in sorted(common):
        delta = after[(task, repeat)] - before[(task, repeat)]
        differences[task].append(delta)
        regressions += delta < 0
        improvements += delta > 0
    result = cluster_interval(differences, draws=draws, seed=seed)
    result.pop("n_observations")
    result.update(n_pairs=len(common), baseline_only=len(before.keys()-after.keys()),
                  treatment_only=len(after.keys()-before.keys()),
                  regressions=regressions, improvements=improvements)
    return result
