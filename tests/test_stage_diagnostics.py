"""Offline stage evidence tests; existing scorer, tasks and prompts untouched."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from stage_diagnostics import diagnose


def call(tool="send_money", **critical):
    return {"tool": tool, "critical": critical or {"recipient": "Alice", "amount": 10}}


def task(*refs):
    return {"id": "fixture", "agent2_calls": list(refs or [call()]),
            "agent2_held_tools": ["send_money", "schedule_transaction"],
            "allowed_tools": ["send_money"], "critical_aliases": {}}


def actual(tool="send_money", **args):
    return {"tool": tool, "args": args or {"recipient": "Alice", "amount": 10}}


class StageEvidence(unittest.TestCase):
    def setUp(self):
        self.no_network = patch("urllib.request.urlopen", side_effect=AssertionError("offline tests only"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def run_case(self, t=None, **kwargs):
        values = {"handoff": "PLAN: call send_money\nEVIDENCE: Alice amount 10",
                  "compressed": "Alice amount 10", "tool_calls": [actual()],
                  "agent2_raw": '{"tool_calls": []}',
                  "condition": "unconditional"}
        values.update(kwargs)
        return diagnose(t or task(), **values)

    def amount(self, result, index=0):
        return next(f for f in result["fields"] if f["argument"] == "amount"
                    and f["required_call_index"] == index)

    def test_correct_output_is_retained_and_inputs_are_not_modified(self):
        t = task()
        saved = copy.deepcopy(t)
        out = self.run_case(t)
        self.assertEqual(out["status"], "analyzed")
        self.assertEqual(out["counts"]["retained"], 2)
        self.assertEqual(t, saved)
        json.dumps(out)

    def test_value_present_before_but_missing_after_is_only_loss_candidate(self):
        out = self.run_case(compressed="Alice", tool_calls=[actual(recipient="Alice", amount=0)])
        field = self.amount(out)
        self.assertEqual(field["status"], "compressor_loss_candidate")
        self.assertTrue(field["literal_loss_observed"])
        self.assertIn("not proven causes", out["limitations"][0])

    def test_upstream_omission_is_distinct_from_compression(self):
        out = self.run_case(handoff="PLAN: call send_money\nAlice", compressed="Alice",
                            tool_calls=[actual(recipient="Alice", amount=0)])
        self.assertEqual(self.amount(out)["status"], "agent1_omission_candidate")
        self.assertEqual(out["counts"]["compressor_loss_candidate"], 0)

    def test_correct_output_prevents_false_loss_even_when_literal_missing(self):
        out = self.run_case(handoff="Alice", compressed="Alice")
        field = self.amount(out)
        self.assertEqual(field["status"], "retained")
        self.assertFalse(field["handoff_presence"]["present"])

    def test_retained_input_and_wrong_argument_is_downstream_candidate(self):
        out = self.run_case(tool_calls=[actual(recipient="Alice", amount=0)])
        self.assertEqual(self.amount(out)["status"], "agent2_failure_candidate")

    def test_none_uses_handoff_not_missing_compressed_text(self):
        out = self.run_case(condition="none", compressed=None, tool_calls=[])
        self.assertEqual(out["status"], "analyzed")
        self.assertEqual(self.amount(out)["status"], "agent2_failure_candidate")
        self.assertEqual(out["counts"]["compressor_loss_candidate"], 0)

    def test_known_empty_transmission_is_not_missing_evidence(self):
        out = self.run_case(compressed="", tool_calls=[])
        self.assertEqual(out["status"], "analyzed")
        self.assertEqual(out["counts"]["compressor_loss_candidate"], 2)
        missing = self.run_case(compressed=None, tool_calls=[])
        self.assertEqual(missing["status"], "missing_evidence")
        self.assertEqual(missing["counts"]["missing_evidence"], 2)

    def test_empty_upstream_is_an_observed_omission(self):
        out = self.run_case(handoff="", compressed="", tool_calls=[])
        self.assertEqual(out["counts"]["agent1_omission_candidate"], 2)

    def test_missing_accepted_log_does_not_become_silent_model_failure(self):
        out = self.run_case(tool_calls=None, agent2_raw='{"tool_calls": []}')
        self.assertEqual(out["status"], "missing_evidence")
        self.assertTrue(all(f["observed_argument_match"] is None for f in out["fields"]))

    def test_empty_log_without_agent2_trace_is_missing_not_a_failure(self):
        out = self.run_case(condition="none", compressed=None, tool_calls=[], agent2_raw=None)
        self.assertEqual(out["status"], "missing_evidence")
        self.assertEqual(out["counts"]["missing_evidence"], 2)
        self.assertTrue(all(f["observed_argument_match"] is None for f in out["fields"]))
        observed = self.run_case(condition="none", compressed=None, tool_calls=[], agent2_raw="")
        self.assertEqual(observed["status"], "analyzed")
        self.assertEqual(observed["counts"]["agent2_failure_candidate"], 2)

    def test_positive_accepted_output_is_evidence_even_without_agent2_raw(self):
        out = self.run_case(agent2_raw=None)
        self.assertEqual(out["counts"]["retained"], 2)

    def test_api_failure_is_not_task_failure_attribution(self):
        out = self.run_case(api_failure=True, tool_calls=[])
        self.assertEqual(out["status"], "api_failure")
        self.assertEqual(out["counts"]["api_failure"], 2)
        self.assertEqual(out["counts"]["agent2_failure_candidate"], 0)

    def test_numeric_substring_is_never_retention_credit(self):
        t = task(call(amount=1, recipient="Alice"))
        out = self.run_case(t, handoff="Alice amount 100", compressed="Alice amount 100",
                            tool_calls=[actual(recipient="Alice", amount=0)])
        field = self.amount(out)
        self.assertIsNone(field["handoff_presence"]["present"])
        self.assertEqual(field["handoff_presence"]["method"], "numeric_substring_match_rejected")
        self.assertEqual(field["status"], "missing_evidence")

    def test_numeric_equivalence_uses_entire_token(self):
        out = self.run_case(handoff="Alice amount 10.00", compressed="Alice amount 10.0")
        self.assertTrue(self.amount(out)["handoff_presence"]["present"])

    def test_date_only_match_does_not_prove_timestamp_survival(self):
        t = task(call("create_calendar_event", start_time="2024-05-19 12:00"))
        out = self.run_case(t, handoff="May 19", compressed="May 19", tool_calls=[])
        self.assertIsNone(out["fields"][0]["handoff_presence"]["present"])
        self.assertEqual(out["fields"][0]["handoff_presence"]["method"], "permissive_findability_only")

    def test_accepted_final_alias_is_visible_and_matches_existing_scorer(self):
        t = task(call(recipient="Spotify", amount=10))
        t["critical_aliases"] = {"recipient": ["Spotify", "XX00ACCOUNT"]}
        out = self.run_case(t, handoff="XX00ACCOUNT amount 10", compressed="XX00ACCOUNT amount 10",
                            tool_calls=[actual(recipient="XX00ACCOUNT", amount=10)])
        recipient = next(f for f in out["fields"] if f["argument"] == "recipient")
        self.assertEqual(recipient["handoff_presence"]["method"], "accepted_alias_literal")
        self.assertTrue(recipient["observed_argument_match"])

    def test_values_from_different_calls_do_not_combine_into_success(self):
        t = task(call(recipient="Alice", amount=1), call(recipient="Bob", amount=2))
        out = self.run_case(t, handoff="Alice 1 Bob 2", compressed="Alice 1 Bob 2",
                            tool_calls=[actual(recipient="Alice", amount=2), actual(recipient="Bob", amount=1)])
        self.assertEqual(len(out["fields"]), 4)
        self.assertTrue(all(f["observed_argument_match"] is None for f in out["fields"]))
        self.assertEqual(out["counts"]["ambiguous_call_alignment_fields"], 4)

    def test_duplicate_requirements_consume_distinct_actual_calls(self):
        out = self.run_case(task(call(), call()), tool_calls=[actual()])
        self.assertEqual(out["counts"]["retained"], 2)
        self.assertEqual(out["counts"]["missing_evidence"], 2)
        self.assertEqual(out["counts"]["agent2_failure_candidate"], 0)

    def test_overlapping_refs_use_a_maximum_complete_assignment(self):
        t = task(call(recipient="Alice"), call(recipient="Alice", amount=10))
        out = self.run_case(t, tool_calls=[actual(recipient="Alice", amount=10),
                                          actual(recipient="Alice", amount=20)])
        self.assertEqual(len(out["fields"]), 3)
        self.assertTrue(all(f["observed_argument_match"] is True for f in out["fields"]))
        self.assertEqual(out["counts"]["agent2_failure_candidate"], 0)

    def test_partial_participant_list_retains_each_surviving_scalar(self):
        t = task(call("create_calendar_event", participants=["Alice", "Bob"]))
        out = self.run_case(t, handoff="Alice Bob", compressed="Alice Bob",
                            tool_calls=[actual("create_calendar_event", participants=["Alice"])])
        fields = {f["expected_scalar"]: f for f in out["fields"]}
        self.assertTrue(fields["Alice"]["observed_argument_match"])
        self.assertEqual(fields["Alice"]["status"], "retained")
        self.assertFalse(fields["Bob"]["observed_argument_match"])
        self.assertEqual(fields["Bob"]["status"], "agent2_failure_candidate")

    def test_partial_nested_attachments_retain_only_the_observed_id(self):
        t = task(call("send_email", attachments=[{"type": "file", "file_id": "19"},
                                                {"type": "file", "file_id": "20"}]))
        out = self.run_case(t, handoff="file 19 file 20", compressed="file 19 file 20",
                            tool_calls=[actual("send_email", attachments=[{"type": "file", "file_id": 19}])])
        ids = {f["expected_scalar"]: f for f in out["fields"] if f["scalar_path"] == "attachments.file_id"}
        self.assertTrue(ids["19"]["observed_argument_match"])
        self.assertEqual(ids["19"]["status"], "retained")
        self.assertFalse(ids["20"]["observed_argument_match"])

    def test_nested_scalar_lists_are_projected_individually(self):
        t = task(call("create_calendar_event", event={"participants": ["Alice", "Bob"]}))
        out = self.run_case(t, handoff="Alice Bob", compressed="Alice Bob",
                            tool_calls=[actual("create_calendar_event", event={"participants": ["Alice"]})])
        fields = {f["expected_scalar"]: f for f in out["fields"]}
        self.assertTrue(fields["Alice"]["observed_argument_match"])
        self.assertFalse(fields["Bob"]["observed_argument_match"])
        self.assertTrue(all(f["scalar_path"] == "event.participants" for f in fields.values()))

    def test_bare_attachment_id_does_not_invent_wrapper_evidence(self):
        t = task(call("send_email", attachments=[{"type": "file", "file_id": "19"},
                                                {"type": "file", "file_id": "20"}]))
        out = self.run_case(t, handoff="file 19 file 20", compressed="file 19 file 20",
                            tool_calls=[actual("send_email", attachments=[19])])
        ids = {f["expected_scalar"]: f for f in out["fields"] if f["scalar_path"] == "attachments.file_id"}
        self.assertTrue(ids["19"]["observed_argument_match"])
        self.assertFalse(ids["20"]["observed_argument_match"])
        self.assertTrue(all(f["observed_argument_match"] is None for f in out["fields"]
                            if f["scalar_path"] == "attachments.type"))

    def test_wrong_tool_gets_no_argument_credit(self):
        out = self.run_case(tool_calls=[actual("schedule_transaction")])
        self.assertTrue(all(f["observed_argument_match"] is False for f in out["fields"]))

    def test_schema_warning_does_not_override_value_in_evidence_or_repaired_action(self):
        t = task(call("update_scheduled_transaction", id=7, amount=1200))
        t["agent2_held_tools"] = t["allowed_tools"] = ["update_scheduled_transaction"]
        out = self.run_case(t, handoff="PLAN: call schedule_transaction with amount=1200\nEVIDENCE: scheduled id=7",
                            compressed="scheduled id=7 amount=1200",
                            tool_calls=[actual("update_scheduled_transaction", id=7, amount=1200)])
        flags = [w for w in out["schema_warnings"] if w["category"] == "upstream_plan_schema_flag"]
        self.assertTrue(flags)
        self.assertFalse(flags[0]["proves_information_loss"])
        self.assertEqual(out["counts"]["retained"], 2)

    def test_plan_counts_and_parser_uncertainty_are_separate_warnings(self):
        out = self.run_case(task(call(), call()), handoff="PLAN:\n1. call send_money and call send_money\nEVIDENCE: Alice 10")
        categories = {w["category"] for w in out["schema_warnings"]}
        self.assertIn("planned_call_count_shortfall", categories)
        self.assertIn("plan_section_or_conjunction_parser_uncertainty", categories)

    def test_configuration_coverage_and_schema_keys_are_separate(self):
        t = task(call("send_direct_message", recipient="Alice", nonexistent_slot="x"))
        out = self.run_case(t, tool_calls=[])
        kinds = {w["category"] for w in out["configuration_warnings"]}
        self.assertIn("required_tools_missing_from_agent2_held_tools", kinds)
        self.assertIn("required_tools_missing_from_allowed_tools", kinds)
        self.assertIn("critical_keys_absent_from_required_tool_schema", kinds)

    def test_current_task_reference_schema_keys_are_valid_without_altering_data(self):
        path = Path(__file__).resolve().parents[1] / "tasks/sample_tasks_all.json"
        tasks = json.loads(path.read_text())
        warned = []
        for t in tasks:
            out = diagnose(t, handoff=None, compressed=None, tool_calls=None, condition="none")
            self.assertFalse(any(w["category"] == "critical_keys_absent_from_required_tool_schema"
                                 for w in out["configuration_warnings"]))
            if any(w["category"] == "required_tools_missing_from_allowed_tools" for w in out["configuration_warnings"]):
                warned.append(t["id"])
        self.assertEqual(set(warned), {"ws_13", "ws_19", "sl_11", "sl_16", "sl_18", "sl_19", "sl_20", "bk_15"})

    def test_empty_reference_set_is_not_applicable(self):
        self.assertEqual(diagnose({"id": "empty", "agent2_calls": []}, condition="none")["status"], "not_applicable")

    def test_empty_scalar_reference_is_not_new_ground_truth(self):
        t = task({"tool": "send_money", "critical": {"date": None}})
        out = diagnose(t, condition="none")
        self.assertEqual(out["fields"][0]["status"], "not_applicable")

    def test_malformed_accepted_log_is_unknown_not_an_observed_model_failure(self):
        out = self.run_case(tool_calls=["not a normalized call"])
        self.assertEqual(out["counts"]["missing_evidence"], 2)
        self.assertIsNone(self.amount(out)["observed_argument_match"])

    def test_plain_raw_refusal_is_a_schema_observation_not_an_api_error(self):
        out = self.run_case(agent2_raw="I cannot complete this task.", tool_calls=[])
        self.assertIn("raw_text_fallback", {w["category"] for w in out["schema_warnings"]})
        self.assertEqual(out["counts"]["api_failure"], 0)


if __name__ == "__main__":
    unittest.main()
