"""
Invariants for the authored banking tasks.

These tasks are not AgentDojo's, so nothing upstream guards them. The things
that would quietly break them: a filler change that shrinks a band below the
point where the PLAN can exceed the budget, an id colliding with the original
51, the savings account leaking into the documents so bk_h10 stops testing
the handoff, and a required value that Agent 1 cannot see.

Reads the built JSON, so it needs no agentdojo. Skips if the file is absent.
"""

import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARD = os.path.join(ROOT, "tasks", "hard_banking.json")
BASE = os.path.join(ROOT, "tasks", "sample_tasks_all.json")

MIN_WRITES = 5
SAVINGS_IBAN = "GB94BARC10201530093459"


def _load(path):
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{os.path.basename(path)} not built")
    with open(path) as fh:
        return json.load(fh)


class HardBanking(unittest.TestCase):

    def setUp(self):
        self.tasks = _load(HARD)

    def test_there_are_some(self):
        self.assertGreaterEqual(len(self.tasks), 8)

    def test_every_task_needs_several_write_calls(self):
        """The whole point. One write call is a find-one lookup, which is the
        shape that produced a 91-token handoff against a 150-token budget."""
        for t in self.tasks:
            with self.subTest(task=t["id"]):
                writes = [c for c in t["required_calls"]
                          if c["tool"] in t["allowed_tools"]]
                self.assertGreaterEqual(len(writes), MIN_WRITES)
                self.assertEqual(len(writes), t["n_writes"])
                self.assertEqual(len(t["agent2_calls"]), t["n_writes"])

    def test_the_plan_alone_cannot_be_a_no_op_at_a_50_token_budget(self):
        """A floor, not a prediction. EVIDENCE adds to this."""
        import metrics
        for t in self.tasks:
            plan = "\n".join(
                "PLAN: call %s with %s" % (
                    c["tool"], "; ".join(f"{k} = {v}"
                                         for k, v in c["critical"].items()))
                for c in t["agent2_calls"])
            n, _ = metrics.count_tokens(plan)
            with self.subTest(task=t["id"]):
                self.assertGreater(n / 50.0, 1.5)

    def test_no_id_collides_with_the_original_set(self):
        base = {t["id"] for t in _load(BASE)}
        for t in self.tasks:
            self.assertNotIn(t["id"], base)

    def test_the_source_field_does_not_claim_to_be_agentdojo(self):
        """Fifty-one AgentDojo tasks plus twelve of ours is not sixty-three
        AgentDojo tasks, and the record is the only thing that stops the paper
        saying otherwise."""
        for t in self.tasks:
            self.assertIn("authored", t["source"])
            self.assertEqual(t.get("task_shape"), "multi_value")

    def test_every_critical_value_is_visible_to_agent_1(self):
        """Agent 2 holds no read tools, so a value Agent 1 never sees is
        unrecoverable and the task is unanswerable by construction."""
        for t in self.tasks:
            hay = (" ".join(t["documents"]) + " " + t["agent1_task"]).lower()
            for call in t["agent2_calls"]:
                for k, v in call["critical"].items():
                    with self.subTest(task=t["id"], key=k):
                        self.assertIn(str(v).lower().rstrip("0").rstrip("."),
                                      hay.replace(",", ""))

    def test_the_savings_account_is_only_in_the_prompt(self):
        """bk_h10's one account number comes from the prompt and nowhere else.
        If it leaks into the documents, Agent 2 could recover it and the task
        stops testing whether the handoff carried it."""
        h10 = [t for t in self.tasks if t["id"] == "bk_h10"]
        if not h10:
            self.skipTest("bk_h10 not in the set")
        t = h10[0]
        self.assertIn(SAVINGS_IBAN, t["agent1_task"])
        self.assertNotIn(SAVINGS_IBAN, " ".join(t["documents"]))
        for c in t["agent2_calls"]:
            self.assertEqual(c["critical"]["recipient"], SAVINGS_IBAN)

    def test_prose_is_never_scored(self):
        for t in self.tasks:
            for c in t["agent2_calls"]:
                self.assertNotIn("subject", c["critical"])

    def test_dates_are_not_scored(self):
        """A model may pass whatever date the tool needs. Scoring one would
        fail correct runs on a format difference, which has happened here
        before."""
        for t in self.tasks:
            for c in t["agent2_calls"]:
                self.assertNotIn("date", c["critical"])

    def test_controls_travel_on_every_task(self):
        for t in self.tasks:
            self.assertTrue(t["safety_constraint"])
            self.assertTrue(t["neutral_control"])
            self.assertTrue(t["neutral_instruction"])

    def test_no_attack_fields_are_invented(self):
        for t in self.tasks:
            self.assertIsNone(t["harmful_call"])
            self.assertIsNone(t["harmful_critical"])


if __name__ == "__main__":
    unittest.main()
