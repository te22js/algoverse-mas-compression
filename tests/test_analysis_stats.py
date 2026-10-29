"""Synthetic observations validate statistics without any model calls."""
import math
import unittest

from analysis_stats import cluster_interval, paired_interval


class ClusterIntervals(unittest.TestCase):
    def test_empty_observations_have_no_estimate_or_interval(self):
        expected = {"estimate": None, "low": None, "high": None,
                    "n_tasks": 0, "n_observations": 0}
        self.assertEqual(cluster_interval({}), expected)
        self.assertEqual(cluster_interval({"empty": []}), expected)

    def test_one_task_reports_mean_without_false_resampling_bounds(self):
        result = cluster_interval({"one": [0, 1, 1]})
        self.assertAlmostEqual(result["estimate"], 2/3)
        self.assertEqual(result["n_tasks"], 1)
        self.assertEqual(result["n_observations"], 3)
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_tasks_have_equal_weight_and_repeats_remain_together(self):
        result = cluster_interval({"many": [0]*20, "one": [1]}, draws=2000)
        self.assertEqual(result["estimate"], .5)
        self.assertEqual(result["low"], 0)
        self.assertEqual(result["high"], 1)
        self.assertEqual(result["n_tasks"], 2)
        self.assertEqual(result["n_observations"], 21)

    def test_reproducibility_includes_mapping_order(self):
        first = {"c": [0, 1], "a": [.2, .3], "b": [.5, .8]}
        second = {key: first[key] for key in reversed(first)}
        self.assertEqual(cluster_interval(first, draws=177, seed=19),
                         cluster_interval(second, draws=177, seed=19))

    def test_nonfinite_or_nonnumeric_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, None, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                cluster_interval({"task": [value]})

    def test_draw_count_must_be_a_positive_integer_even_for_empty_data(self):
        for draws in (0, -1, 1.5, True):
            with self.subTest(draws=draws), self.assertRaises(ValueError):
                cluster_interval({}, draws=draws)
        self.assertIsNotNone(cluster_interval({"a": [0], "b": [1]}, draws=1)["low"])


class PairedIntervals(unittest.TestCase):
    def test_empty_pairs_and_unmatched_coverage(self):
        result = paired_interval({("a", 0): 1}, {("b", 0): 0})
        self.assertEqual(result, {"estimate": None, "low": None, "high": None,
            "n_tasks": 0, "n_pairs": 0, "baseline_only": 1,
            "treatment_only": 1, "regressions": 0, "improvements": 0})

    def test_deltas_direction_counts_and_partial_task_repeats(self):
        baseline = {("a", 0): 1, ("a", 1): 1, ("b", 0): 0,
                    ("c", 0): 1}
        treatment = {("a", 0): 0, ("a", 1): 1, ("b", 0): 1,
                     ("d", 0): 1}
        result = paired_interval(baseline, treatment)
        # Mean task deltas: a=-0.5, b=+1; not the pooled-pair mean 0.
        self.assertEqual(result["estimate"], .25)
        self.assertEqual(result["low"], -.5)
        self.assertEqual(result["high"], 1)
        self.assertEqual(result["n_tasks"], 2)
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["baseline_only"], 1)
        self.assertEqual(result["treatment_only"], 1)
        self.assertEqual(result["regressions"], 1)
        self.assertEqual(result["improvements"], 1)

    def test_single_matched_task_bounds_are_unavailable(self):
        result = paired_interval({("a", 0): 1}, {("a", 0): 0})
        self.assertEqual(result["estimate"], -1)
        self.assertIsNone(result["low"])
        self.assertIsNone(result["high"])

    def test_unmatched_nonfinite_values_are_still_rejected(self):
        for before, after in (({("a", 0): math.nan}, {}),
                              ({}, {("b", 0): math.inf})):
            with self.assertRaises(ValueError):
                paired_interval(before, after)

    def test_paired_seed_and_input_order_are_reproducible(self):
        baseline = {(task, rep): 0 for task in ("c", "b", "a") for rep in (1, 0)}
        treatment = {key: i/10 for i, key in enumerate(baseline)}
        self.assertEqual(paired_interval(baseline, treatment, draws=111, seed=17),
                         paired_interval(dict(reversed(list(baseline.items()))),
                                         dict(reversed(list(treatment.items()))),
                                         draws=111, seed=17))


if __name__ == "__main__":
    unittest.main()
