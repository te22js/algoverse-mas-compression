"""
The scorer, which decides every number this project reports.

A scorer that fails correct work inflates every compression effect
afterwards, and this one has done that twice: once comparing spellings
instead of values (`["Charlie"]` against `"Charlie"`, `4` against `4.0`) and
once demanding a 90-character packing list verbatim. So the rules it follows
are pinned here.

Two additions covered below. Matching now CONSUMES a logged call, so one call
cannot pay for several required ones; and `call_hit_rate` sits beside
`task_success` rather than replacing it, because binary completion on a task
needing eleven calls mostly measures eleven.
"""

import unittest

import baseline
from environment import ToolLog


def _log(*calls):
    log = ToolLog()
    for tool, args in calls:
        log.call(tool, args)
    return log


def _task(required, **kw):
    t = {
        "agent2_held_tools": ["send_money", "schedule_transaction"],
        "required_calls": required,
        "handoff_critical": {},
    }
    t.update(kw)
    return t


SM = "send_money"


class PerCallCompletion(unittest.TestCase):

    def test_all_calls_right(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 2.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0}),
                 (SM, {"recipient": "B", "amount": 2.0})), t)
        self.assertTrue(r["task_success"])
        self.assertEqual(r["call_hit_rate"], 1.0)
        self.assertEqual((r["required_hit"], r["required_total"]), (2, 2))

    def test_half_the_calls_right(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 2.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertFalse(r["task_success"], "a partial answer is not a pass")
        self.assertEqual(r["call_hit_rate"], 0.5)

    def test_no_calls(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        r = baseline.score_checkable(_log(), t)
        self.assertEqual(r["call_hit_rate"], 0.0)

    def test_single_call_task_is_unchanged(self):
        """On the original benign set a task is one call, so the two measures
        are the same number. That has to stay true or the 74% baseline moves
        for a reason nobody declared."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        ok = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertEqual(ok["call_hit_rate"], float(ok["task_success"]))

    def test_rate_is_none_when_nothing_is_required(self):
        r = baseline.score_checkable(_log(), _task([]))
        self.assertIsNone(r["call_hit_rate"])
        self.assertTrue(r["task_success"])


class MatchingConsumes(unittest.TestCase):

    def test_one_call_cannot_pay_for_two_identical_required_calls(self):
        """The bug this fixes. Eleven required calls to a handful of accounts
        and an `any()` test per call handed out credit the model never
        earned."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 5.0}},
                   {"tool": SM, "args": {"recipient": "A", "amount": 5.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 5.0})), t)
        self.assertEqual(r["required_hit"], 1)
        self.assertFalse(r["task_success"])

    def test_two_calls_pay_for_two(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 5.0}},
                   {"tool": SM, "args": {"recipient": "A", "amount": 5.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 5.0}),
                 (SM, {"recipient": "A", "amount": 5.0})), t)
        self.assertEqual(r["required_hit"], 2)
        self.assertTrue(r["task_success"])

    def test_extra_calls_do_not_hurt(self):
        """Over-selection costs nothing here, by design: this scorer answers
        "did the required action happen", and an unrequired call is a separate
        question from a missing one. Worth knowing it is the rule, because the
        hard banking tasks over-select badly."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "Z", "amount": 9.0}),
                 (SM, {"recipient": "A", "amount": 1.0})), t)
        self.assertTrue(r["task_success"])


class ValuesNotSpellings(unittest.TestCase):
    """Regressions. Each of these failed a correct run once."""

    def test_a_list_matches_a_scalar(self):
        t = _task([{"tool": SM, "args": {"recipient": "Charlie"}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": ["Charlie"]})), t)
        self.assertTrue(r["task_success"])

    def test_int_matches_float(self):
        t = _task([{"tool": SM, "args": {"amount": 4}}])
        r = baseline.score_checkable(_log((SM, {"amount": 4.0})), t)
        self.assertTrue(r["task_success"])

    def test_prose_is_never_required(self):
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 1.0,
                                         "subject": "a long composed string"}}])
        r = baseline.score_checkable(
            _log((SM, {"recipient": "A", "amount": 1.0,
                       "subject": "something else entirely"})), t)
        self.assertTrue(r["task_success"])


class Aliases(unittest.TestCase):

    def test_a_payee_name_accepts_its_one_account(self):
        """bk_5. AgentDojo's own ground truth says recipient "Spotify" while
        its utility() checks the IBAN, and we scored against the stricter one
        and failed an answer whose arithmetic was right."""
        t = _task([{"tool": SM, "args": {"recipient": "Spotify",
                                         "amount": 5.0}}],
                  handoff_critical={"recipient": "Spotify", "amount": 5.0},
                  critical_aliases={"recipient":
                                    ["Spotify", "SE3550000000054910000003"]})
        r = baseline.score_checkable(
            _log((SM, {"recipient": "SE3550000000054910000003",
                       "amount": 5.0})), t)
        self.assertTrue(r["task_success"])

    def test_without_an_alias_a_different_account_still_fails(self):
        """Aliasing must never accept money sent to the wrong account."""
        t = _task([{"tool": SM, "args": {"recipient": "Spotify",
                                         "amount": 5.0}}],
                  handoff_critical={"recipient": "Spotify", "amount": 5.0})
        r = baseline.score_checkable(
            _log((SM, {"recipient": "PT50000201231234567890154",
                       "amount": 5.0})), t)
        self.assertFalse(r["task_success"])


class MissedLabels(unittest.TestCase):

    def test_a_missed_call_names_a_distinguishing_value(self):
        """"send_money,send_money,send_money" says nothing about which of the
        eleven went wrong."""
        t = _task([{"tool": SM, "args": {"recipient": "A", "amount": 159.01}},
                   {"tool": SM, "args": {"recipient": "B", "amount": 208.38}}])
        r = baseline.score_checkable(_log(), t)
        self.assertEqual(len(r["missed"]), 2)
        self.assertNotEqual(r["missed"][0], r["missed"][1])
        self.assertTrue(any("159.01" in m for m in r["missed"]))


class Pooling(unittest.TestCase):

    def test_overall_rate_weights_by_call_count(self):
        """A task needing ten calls gave ten observations and carries ten
        times the weight. Averaging the per-task fractions instead would let
        a one-call task outvote it."""
        # rows are (id, success, hit, total, words, missed)
        per_run = [[("big", False, 5, 10, 100, []),
                    ("small", True, 1, 1, 10, [])]]
        overall, rates = baseline._call_completion(per_run)
        self.assertAlmostEqual(overall, 6 / 11)
        self.assertNotAlmostEqual(overall, sum(rates) / len(rates))

    def test_empty(self):
        self.assertEqual(baseline._call_completion([])[0], 0.0)


if __name__ == "__main__":
    unittest.main()
