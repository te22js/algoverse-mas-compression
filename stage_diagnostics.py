"""Supplemental stage evidence, never a replacement scorer or causal oracle.

PLAN schema warnings describe named operations; they do not prove that a value
was lost from free-text EVIDENCE. All rows stay in the primary denominator.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import re

import baseline
import failure_stage
import obtainable


STATUSES = ("api_failure", "missing_evidence", "agent1_omission_candidate",
            "compressor_loss_candidate", "agent2_failure_candidate", "retained",
            "not_applicable")
LIMITATIONS = [
    "Candidates are observational evidence, not proven causes; multiple stages can contribute.",
    "Schema warnings never exclude a row, change a score, or establish information loss.",
    "Literal presence can be incidental or negated; absence can be paraphrase or calculation.",
    "A correct output argument takes precedence over lexical absence; it does not prove textual retention.",
    "Scalar output matches do not establish whole-list equality, record associations, or complete call correctness.",
    "Partial multi-call alignment is unknown when values could belong to different calls.",
    "Independent upstream draws cannot isolate compression effects without identical handoff evidence.",
    "Only existing agent2_calls critical references are examined; this is not complete task utility.",
]
_NUMBER = re.compile(r"(?<![\w.+-])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?![\w.])")


def _decimal(value):
    try:
        result = Decimal(str(value).replace(",", ""))
        return result if result.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def _presence(text, key, value, aliases=()):
    if text is None:
        return {"present": None, "method": "missing_text",
                "legacy_findable": None, "legacy_method": None}
    hay = obtainable.norm(text)
    legacy, legacy_method = obtainable.findable(key, value, hay)
    target = obtainable.norm(value)
    result = {"present": None, "method": "unresolved", "legacy_findable": legacy,
              "legacy_method": legacy_method, "low_specificity": len(target) <= 2}
    if value is None or target in ("", "none", "[]", "{}"):
        return {**result, "method": "empty_reference_not_applicable"}
    numeric = _decimal(value) if not isinstance(value, bool) else None
    if numeric is not None:
        found = any(_decimal(m.group(0)) == numeric for m in _NUMBER.finditer(hay))
        return {**result, "present": True if found else None if legacy else False,
                "method": "numeric_token" if found else
                          "numeric_substring_match_rejected" if legacy else "absent_numeric_token"}
    for candidate in (value, *aliases):
        literal = obtainable.norm(candidate)
        if literal and re.search(r"(?<!\w)" + re.escape(literal) + r"(?!\w)", hay):
            return {**result, "present": True,
                    "method": "bounded_literal" if candidate == value else "accepted_alias_literal"}
    # The team's findability allows date-only matches for timestamps and word
    # fragments across records. Keep that evidence, but do not call it retained.
    return {**result, "present": None if legacy else False,
            "method": "permissive_findability_only" if legacy else "no_literal_or_legacy_match"}


def _match(args, expected, aliases=None):
    try:
        return baseline._args_match(args, expected, aliases)
    except (TypeError, ValueError, AttributeError):
        return None


def _aliases(task, index, refs):
    return (task.get("critical_aliases") or {}) if index == len(refs) - 1 else {}


def _leaves(critical):
    """Extend the team's one-level flatten to nested scalar leaves."""
    for path, value in obtainable.flatten(critical):
        if isinstance(value, dict):
            yield from ((path + "." + child, scalar) for child, scalar in _leaves(value))
        elif isinstance(value, (list, tuple)):
            for child, scalar in _leaves({path: value}):
                yield child, scalar
        else:
            yield path, value


def _align(refs, calls, task):
    """Maximize complete call matches, then bind only unique residual calls.

    This diagnostic alignment does not reproduce the primary scorer's greedy
    order: an overlapping loose reference must not steal a tighter one's only
    complete match and manufacture a downstream failure candidate.
    """
    if calls is None:
        return {}, [], "missing_accepted_call_log"
    clean = [c for c in calls if isinstance(c, dict) and isinstance(c.get("tool"), str)
             and isinstance(c.get("args"), dict)]
    edges = {ri: [ci for ci, actual in enumerate(clean)
                  if actual["tool"] == ref["tool"] and _match(
                      actual["args"], ref.get("critical") or {}, _aliases(task, ri, refs)) is True]
             for ri, ref in enumerate(refs)}
    owner = {}

    def augment(ri, seen):
        for ci in edges[ri]:
            if ci in seen:
                continue
            seen.add(ci)
            if ci not in owner or augment(owner[ci], seen):
                owner[ci] = ri
                return True
        return False

    for ri in range(len(refs)):
        augment(ri, set())
    matches = {ri: (ci, "complete_reference_match") for ci, ri in owner.items()}
    used = set(owner)
    residual = Counter(ref["tool"] for ri, ref in enumerate(refs) if ri not in matches)
    for ri, ref in enumerate(refs):
        if ri in matches or edges[ri]:
            continue
        candidates = [ci for ci, c in enumerate(clean) if ci not in used and c["tool"] == ref["tool"]]
        if len(candidates) == 1 and residual[ref["tool"]] == 1:
            matches[ri] = (candidates[0], "unique_partial_call")
    return matches, clean, None if len(clean) == len(calls) else "malformed_accepted_call_log_entries"


def _scalar_match(args, argument, path, scalar, reference, aliases):
    """Project a flattened leaf, preserving the team's scalar normalizations.

    Lists are unordered in the primary matcher. For this supplemental evidence
    only, test membership of each required scalar separately so a missing Bob
    cannot erase an observed Alice. Nested dict leaves are read by key. A bare
    ID may stand for the sole ID leaf of an attachment wrapper, as in baseline;
    other missing wrapper fields remain unknown rather than invented.
    """
    if argument not in args:
        return False
    tail = path.split(".")[1:]
    expected = reference[argument]
    got = args[argument]
    if not tail and not isinstance(expected, (list, tuple)):
        return _match({argument: got}, {argument: scalar}, aliases)

    def project(value, parts):
        if isinstance(value, (list, tuple)):
            answers = [project(item, parts) for item in value]
            return True if True in answers else None if None in answers else False
        if not parts:
            return _match({argument: value}, {argument: scalar}, aliases)
        if isinstance(value, dict):
            return project(value[parts[0]], parts[1:]) if parts[0] in value else False
        expected_records = expected if isinstance(expected, (list, tuple)) else [expected]
        # baseline._norm collapses a dict with exactly one *_id/id key to
        # its ID. That supports evidence for the ID, never its type wrapper.
        if len(parts) == 1 and any(
                isinstance(record, dict)
                and [key for key in record if key.endswith("_id") or key == "id"] == parts
                for record in expected_records):
            return _match({parts[0]: value}, {parts[0]: scalar})
        return None

    return project(got, tail)


def _output_argument(refs, ri, argument, path, scalar, task, matches, calls, log_missing):
    if log_missing:
        return None, "missing_accepted_call_log"
    ref = refs[ri]
    aliases = _aliases(task, ri, refs)
    compare = lambda args: _scalar_match(args, argument, path, scalar, ref["critical"], aliases)
    if ri in matches:
        index, kind = matches[ri]
        return compare(calls[index]["args"]), kind
    used = {index for index, _ in matches.values()}
    if any(ci in used and c["tool"] == ref["tool"] and _match(
            c["args"], ref.get("critical") or {}, aliases) is True
           for ci, c in enumerate(calls)):
        return None, "ambiguous_overlapping_required_calls"
    candidates = [c for ci, c in enumerate(calls) if ci not in used and c["tool"] == ref["tool"]]
    if not candidates:
        return False, "no_remaining_same_tool_call"
    possibilities = [compare(c["args"]) for c in candidates]
    if all(value is False for value in possibilities):
        return False, "all_residual_same_tool_calls_mismatch"
    return None, "ambiguous_partial_multicall_alignment"


def _warnings(task, refs, handoff, raw_by_stage):
    schema, config = [], []
    required = Counter(ref["tool"] for ref in refs)
    held, allowed = task.get("agent2_held_tools"), task.get("allowed_tools")
    for key, tools in (("agent2_held_tools", held), ("allowed_tools", allowed)):
        if tools is None:
            config.append({"category": "tool_set_unavailable", "tool_set": key})
        elif set(required) - set(tools):
            config.append({"category": "required_tools_missing_from_" + key,
                           "tools": sorted(set(required) - set(tools))})
    for ri, ref in enumerate(refs):
        params = baseline.TOOL_PARAMS.get(ref["tool"])
        if params is None:
            config.append({"category": "required_tool_schema_unknown", "required_call_index": ri,
                           "tool": ref["tool"]})
        elif set(ref.get("critical") or {}) - set(params):
            config.append({"category": "critical_keys_absent_from_required_tool_schema",
                           "required_call_index": ri, "tool": ref["tool"],
                           "keys": sorted(set(ref.get("critical") or {}) - set(params))})
    if handoff is not None:
        named = failure_stage.plan_tools(handoff)
        for ri, mismatch in enumerate(failure_stage.check(task, handoff, baseline.TOOL_PARAMS)):
            if mismatch.status != failure_stage.EXPRESSIBLE:
                schema.append({"category": "upstream_plan_schema_flag", "required_call_index": ri,
                               **mismatch.as_dict(), "proves_information_loss": False})
        for tool, count in required.items():
            actual = named.count(tool)
            if actual < count:
                schema.append({"category": "planned_call_count_shortfall", "tool": tool,
                               "required_count": count, "parsed_plan_count": actual,
                               "parser_uncertainty": True})
        if (re.search(r"(?im)^\s*PLAN\s*:\s*$", handoff)
                or any(len(re.findall(r"\bcall\s+[a-z_]\w*", line, re.I)) > 1
                       for line in handoff.splitlines() if re.match(r"\s*PLAN\b", line, re.I))):
            schema.append({"category": "plan_section_or_conjunction_parser_uncertainty"})
    for stage, raw in raw_by_stage.items():
        if raw is not None:
            try:
                parsed = baseline._parse_tolerant(raw)
                calls = parsed.get("tool_calls", [])
                if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
                    schema.append({"category": "raw_tool_calls_shape_uncertain", "stage": stage})
                elif raw.strip() and not calls and parsed.get("handoff_note") == raw.strip():
                    schema.append({"category": "raw_text_fallback", "stage": stage})
            except (TypeError, ValueError, AttributeError):
                schema.append({"category": "raw_parse_uncertain", "stage": stage})
    return schema, config


def _status(before, after, observed, condition, api_failure, applicable):
    if not applicable:
        return "not_applicable"
    if api_failure:
        return "api_failure"
    if observed is True:
        return "retained"
    if before is None or after is None or observed is None:
        return "missing_evidence"
    if before is False:
        return "agent1_omission_candidate"
    if after is False and condition != "none":
        return "compressor_loss_candidate"
    return "agent2_failure_candidate"


def diagnose(task, *, agent1_raw=None, handoff=None, compressed=None,
             agent2_raw=None, tool_calls=None, condition, api_failure=False):
    """Analyze existing traces without calling models or modifying task/score.

    None is unavailable evidence; an empty raw string is observed silence. An
    empty accepted log alone does not establish that Agent 2 ran. `tool_calls`
    is the accepted execution log, never silently replaced
    by calls extracted from raw prose. `none` transmits the original handoff.
    """
    refs = task.get("agent2_calls") or []
    schema, config = _warnings(task, refs, handoff,
                               {"agent1": agent1_raw, "agent2": agent2_raw})
    calls_valid = tool_calls is None or isinstance(tool_calls, list)
    matches, calls, alignment_warning = _align(refs, tool_calls if calls_valid else None, task)
    if alignment_warning:
        schema.append({"category": alignment_warning})
    if condition == "none" and compressed is not None and compressed != handoff:
        config.append({"category": "none_condition_has_different_compressed_text"})
    transmitted = handoff if condition == "none" else compressed
    agent2_observed = agent2_raw is not None or bool(calls)
    fields = []
    for ri, ref in enumerate(refs):
        for path, scalar in _leaves(ref.get("critical") or {}):
            argument = path.split(".", 1)[0]
            aliases = _aliases(task, ri, refs).get(argument, ()) if path == argument else ()
            before, after = _presence(handoff, path, scalar, aliases), _presence(transmitted, path, scalar, aliases)
            observed, alignment = _output_argument(refs, ri, argument, path, scalar, task, matches, calls,
                                                   tool_calls is None or not calls_valid)
            if alignment_warning == "malformed_accepted_call_log_entries" and observed is False:
                observed, alignment = None, "malformed_accepted_call_log_entries"
            if observed is not True and not agent2_observed:
                observed, alignment = None, "missing_agent2_trace"
            applicable = scalar is not None and obtainable.norm(scalar) not in ("", "none", "[]", "{}")
            fields.append({"required_call_index": ri, "required_tool": ref["tool"],
                           "argument": argument, "scalar_path": path, "expected_scalar": scalar,
                           "handoff_presence": before, "transmitted_presence": after,
                           "observed_argument_match": observed, "output_alignment": alignment,
                           "literal_loss_observed": before["present"] is True and after["present"] is False,
                           "status": _status(before["present"], after["present"], observed,
                                             condition, api_failure, applicable)})
    counts = {key: sum(f["status"] == key for f in fields) for key in STATUSES}
    counts.update(fields_total=len(fields), schema_warning_count=len(schema),
                  configuration_warning_count=len(config),
                  uncertain_presence_fields=sum(any(f[k]["present"] is None for k in
                                                     ("handoff_presence", "transmitted_presence")) for f in fields),
                  ambiguous_call_alignment_fields=sum(f["output_alignment"].startswith("ambiguous_") for f in fields))
    status = "api_failure" if api_failure else "not_applicable" if not fields else (
        "missing_evidence" if handoff is None or transmitted is None or tool_calls is None or not calls_valid
        or not agent2_observed else "analyzed")
    return {"status": status, "schema_warnings": schema, "configuration_warnings": config,
            "fields": fields, "counts": counts, "limitations": list(LIMITATIONS)}
