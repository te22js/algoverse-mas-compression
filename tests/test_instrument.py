"""
The assumptions the instrumentation rests on, asserted rather than hoped.

Stage labelling reads the system prompt to tell Agent 1 from Agent 2. That is
a prose assumption, and prose gets rewritten — the Agent 1 prompt has already
been rewritten five times in this project. If someone changes its opening
line, these tests fail; without them, every trace line from Agent 1 would
quietly be filed as a compressor call and the failure taxonomy built on top
would be wrong in a way no score would reveal.

Also here: end-to-end metrics through the real patched pipeline, with a fake
backend. No API calls.
"""

import unittest

import baseline            # installs every patch on import
import agent1_prompts
import instrument
import pipeline


class PromptPrefixContract(unittest.TestCase):

    def test_every_agent1_regime_announces_itself(self):
        for name, text in agent1_prompts.REGIMES.items():
            with self.subTest(regime=name):
                self.assertTrue(
                    text.startswith(instrument.AGENT1_PREFIX),
                    f"regime {name!r} no longer starts with "
                    f"{instrument.AGENT1_PREFIX!r}, so its calls would be "
                    f"traced as compressor calls")

    def test_the_brief_regime_too(self):
        self.assertTrue(
            baseline._BRIEF_SYSTEM.startswith(instrument.AGENT1_PREFIX))

    def test_agent2_template_announces_itself(self):
        self.assertTrue(
            baseline._AGENT2_TEMPLATE.startswith(instrument.AGENT2_PREFIX))

    def test_the_two_prefixes_cannot_both_match(self):
        self.assertFalse(
            instrument.AGENT2_PREFIX.startswith(instrument.AGENT1_PREFIX))

    def test_classifier(self):
        f = instrument.stage_from_system
        self.assertEqual(f("You are Agent 1 in a two-agent workflow."),
                         instrument.STAGE_AGENT1)
        self.assertEqual(f("You are Agent 2 in a two-agent workflow."),
                         instrument.STAGE_AGENT2)
        self.assertEqual(f("You are a summarizer. Compress this."),
                         instrument.STAGE_COMPRESSOR)
        self.assertEqual(f(""), instrument.STAGE_COMPRESSOR)
        self.assertEqual(f(None), instrument.STAGE_COMPRESSOR)


TASK = {
    "id": "fake_bk", "domain": "banking",
    "agent1_task": "Send the difference for the March payment.",
    "documents": ["Transaction(id=3, subject='Spotify Premium') " * 200],
    "agent1_tools": ["get_most_recent_transactions"],
    "agent2_held_tools": ["send_money"],
    "required_calls": [{"tool": "send_money",
                        "args": {"recipient": "XX00", "amount": 5.0}}],
    "handoff_critical": {"recipient": "XX00", "amount": 5.0},
}


class _Fake:
    """Agent 1 writes a long handoff; Agent 2 makes the right call."""

    name = "fake"

    def __init__(self, handoff="word " * 400):
        self.handoff = handoff

    def chat(self, system, user):
        import json
        if system.startswith(instrument.AGENT1_PREFIX):
            return json.dumps({"tool_calls": [],
                               "handoff_note": self.handoff})
        if system.startswith(instrument.AGENT2_PREFIX):
            return json.dumps({"tool_calls": [
                {"tool": "send_money",
                 "args": {"recipient": "XX00", "amount": 5.0}}],
                "handoff_note": "done"})
        return "a compressed handoff"            # the compressor


class EndToEnd(unittest.TestCase):
    """Through the real patched pipeline, with the money turned off."""

    def setUp(self):
        baseline.RUN = None
        baseline.TRACE = None

    def run_one(self, condition="none", backend=None, budget=150):
        import config
        baseline.RUN = config.RunConfig(condition=condition,
                                        compressor_budget_tokens=budget)
        b = backend or _Fake()
        return pipeline.run_pipeline(TASK, b, b, condition, repeat=0)

    def test_metrics_are_attached_to_every_run(self):
        m = self.run_one()["metrics"]
        for k in ("source_context_tokens", "agent1_input_tokens",
                  "precompression_handoff_tokens", "compressor_budget_tokens",
                  "compressed_tokens", "cer", "realized_ratio",
                  "fraction_removed", "compression_stratum",
                  "token_count_method", "condition", "compression_applied"):
            self.assertIn(k, m)

    def test_condition_none_is_recorded_as_no_compression(self):
        m = self.run_one("none")["metrics"]
        self.assertEqual(m["condition"], "none")
        self.assertFalse(m["compression_applied"])
        self.assertIsNone(m["compressed_tokens"])
        self.assertIsNotNone(m["cer"], "exposure is still measurable")

    def test_source_context_is_the_documents_not_the_handoff(self):
        m = self.run_one()["metrics"]
        self.assertGreater(m["source_context_tokens"],
                           m["precompression_handoff_tokens"])

    def test_agent1_input_exceeds_the_documents_by_the_task_statement(self):
        m = self.run_one()["metrics"]
        self.assertGreater(m["agent1_input_tokens"],
                           m["source_context_tokens"])

    def test_a_small_handoff_is_labelled_unexposed(self):
        """bk_5's shape: 27 words into a 150-token budget. Compression is not
        weak on this task, it is a no-op, and the record has to say so."""
        m = self.run_one(backend=_Fake("PLAN: send_money, amount 5.0"))["metrics"]
        self.assertLess(m["cer"], 1.0)
        self.assertEqual(m["compression_stratum"], "unexposed")

    def test_a_large_handoff_is_labelled_exposed(self):
        m = self.run_one(backend=_Fake("word " * 2000))["metrics"]
        self.assertGreater(m["cer"], 3.0)
        self.assertEqual(m["compression_stratum"], "strong")

    def test_budget_changes_the_cer_and_is_recorded(self):
        a = self.run_one(budget=150)["metrics"]
        b = self.run_one(budget=50)["metrics"]
        self.assertEqual(a["precompression_handoff_tokens"],
                         b["precompression_handoff_tokens"])
        self.assertEqual(b["compressor_budget_tokens"], 50)
        self.assertGreater(b["cer"], a["cer"])

    def test_every_model_call_is_traced(self):
        r = self.run_one()
        stages = [c["who"] for c in r["text"]["raw"]]
        self.assertEqual(stages, ["agent1", "agent2"],
                         "condition=none makes no compressor call")

    def test_the_compressor_call_is_traced_separately(self):
        r = self.run_one("unconditional")
        stages = [c["who"] for c in r["text"]["raw"]]
        self.assertEqual(stages, ["agent1", "compressor", "agent2"])

    def test_realized_compression_is_measured_when_a_compressor_runs(self):
        m = self.run_one("unconditional")["metrics"]
        self.assertTrue(m["compression_applied"])
        self.assertIsNotNone(m["compressed_tokens"])
        self.assertIsNotNone(m["realized_ratio"])

    def test_parse_status_is_recorded(self):
        self.assertEqual(self.run_one()["parse_events"], ["ok", "ok"])

    def test_broken_json_is_recorded_as_repaired_not_as_silence(self):
        """The 32 lost runs. A raw newline inside a JSON string made the
        object invalid, _parse returned no tool calls, and the run was scored
        as a refusal when the model had answered correctly."""
        class Broken(_Fake):
            def chat(self, system, user):
                if system.startswith(instrument.AGENT2_PREFIX):
                    return ('{"tool_calls": [{"tool": "send_money", "args": '
                            '{"recipient": "XX00", "amount": 5.0}}], '
                            '"handoff_note": "line one\nline two"}')
                return super().chat(system, user)

        r = self.run_one(backend=Broken())
        self.assertIn("repaired", r["parse_events"])
        self.assertTrue(r["task_success"],
                        "the repair exists so a correct answer still scores")

    def test_scoring_still_works(self):
        self.assertTrue(self.run_one()["task_success"])

    def test_runs_without_a_config_do_not_crash(self):
        """selftest replays ground truth with RUN unset."""
        baseline.RUN = None
        b = _Fake()
        r = pipeline.run_pipeline(TASK, b, b, "none")
        self.assertIn("metrics", r)


class Audit(unittest.TestCase):

    def test_the_audit_lists_the_one_remaining_per_task_global(self):
        a = instrument.SHARED_STATE_AUDIT
        self.assertEqual(a["per_task_globals"], ["pipeline.AGENT2_SYSTEM"])

    def test_agent2_system_still_answers_per_thread(self):
        """If someone replaces _PerThreadSystem with a plain string, the race
        comes back. The type is the invariant."""
        self.assertTrue(hasattr(pipeline.AGENT2_SYSTEM, "format"))


if __name__ == "__main__":
    unittest.main()
