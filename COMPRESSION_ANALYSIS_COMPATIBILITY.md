# Compression redesign analysis compatibility

This contribution adds offline analysis to the team's existing experiment outputs. It preserves execution, primary scoring, task files, prompts, controls, and compression settings. Its scientific purpose is to distinguish observed upstream omissions, facts lost across the compressor boundary, and downstream execution failures without treating every failed task as a compression effect.

The comparison uses PR [4](https://github.com/yiboyang-sas/algoverse-mas-compression/pull/4) at `b6c5eb66d3380aa5bd1613fc5caff0673bd151a8` and the live branch **`compression-redesign`** at `52958c146d849aa60f50d9c3a78d4e54ecd6f474`. The requested name `compression_redesign` is not a remote branch. The new work starts from the existing hyphenated branch.

## Compatibility and reuse

| Component | Earlier PR 4 | Current team implementation | Adaptation |
|---|---|---|---|
| Evaluation set | Frozen 18 workspace tasks | Explicit `tasks/sample_tasks_63.json`: 18 workspace, 15 Slack, 24 banking, 6 travel | Require an explicit task snapshot; do not pool the benchmarks |
| Model and handoff | Provisional GPT-4o, temperature 0.7, delegation; independent or optional shared handoffs | `baseline.py`, default GPT-4o-mini and temperature 0, `MAS_AGENT1=plan`; Agent 2 sees the transmitted handoff only | Read saved configuration; do not port old switches or add a new runner |
| Compression | Earlier word and character limits | Default nominal 150 tokens; prompted conditions request about 111 words; sliding window retains 600 characters | Report recorded lengths and budget overruns; do not enforce a different budget |
| Primary scoring | Earlier `environment.py` scorer | `baseline.score_checkable`, write calls and critical arguments, distinct call consumption | Read saved scores; reuse its argument matcher only for supplemental evidence checks |
| Tracing | PR 4 journals and shared-handoff artifacts | `instrument.py`, `tracing.py`, `config.json`, `calls.jsonl`, `runs.jsonl`, rich `results.json` | Consume the existing streams; add no model wrapper or trace format |
| Exposure and controls | Earlier token and retention analysis | `metrics.py`, `controls.py`, `sweep.py`, mechanical and oracle rehearsals | Preserve existing metrics; do not rebuild the budget sweep or control detector |
| Failure mechanism | Earlier heuristic attribution | `failure_stage.py` checks PLAN tools against parameter slots | Retain that warning separately from observed field transitions |
| Uncertainty | Task-level bootstrap and paired differences | No task-cluster or paired-outcome interval implementation | Adapt PR 4's percentile and task-cluster bootstrap mathematics in `analysis_stats.py` |

The 63-task file SHA256 is `cc9aed2091594034d68c7205225b810179139f1993cde9a11df4f241f22b51a9`. It contains the 51-task multi-suite set plus 12 authored hard banking tasks. `baseline.py` currently defaults to the separate 14-task `sample_tasks.json`; `RunConfig.tasks_file` defaults to the 51-task file. Therefore neither default establishes a 63-task experiment. The analyzer requires `--tasks` and checks the selected IDs, recorded task path, and historical Git task blob where available.

## Missing functionality added

`analyze_redesign.py` reads one existing run directory per condition and joins calls to outcomes using **run ID, task ID, repeat, and stage**. It rejects duplicate outcomes, ambiguous stage records, malformed JSONL, unexpected tasks/repeats, incompatible configurations, mixed token methods, and contradictory handoff boundaries. Missing planned outcomes and orphan call groups remain visible in coverage. It does not infer repeat IDs from `results.json` array order; that file lacks explicit repeat identity.

For each task run, the original handoff is recovered using the team's existing tolerant parser. The transmitted handoff is read from the exact Agent 2 input. Prompted compressor inputs and outputs are checked against those boundaries; mechanical sliding-window output is checked using the existing pure helper. Missing text remains unknown; an observed empty string remains empty. No model compressor is called during analysis.

`stage_diagnostics.py` examines the existing critical references across **all** `agent2_calls`, including multi-call banking tasks. Each scalar records its presence before and after compression, the matching method, a separate legacy findability result, and accepted same-tool output argument matching. Complete call matches consume distinct logged actions using maximum matching, so overlapping references do not arbitrarily blame Agent 2. This supplemental alignment can differ from the primary scorer's greedy matching; it never replaces that saved score. Partial list members are checked individually, and ambiguous partial multi-call alignment remains unknown. Numeric substring matches such as `1` inside `100` do not receive literal-retention credit.

The statuses are supplemental candidates:

- **Agent 1 omission candidate:** a required value is absent before compression and the observed action does not satisfy it.
- **Compressor loss candidate:** a value is observed before compression, absent in the transmitted text, and not recovered in the action.
- **Agent 2 failure candidate:** required values remain observable, but the accepted same-tool argument is wrong or no corresponding action remains.
- **Retained:** the accepted output argument matches the reference. This can include inference or recovery; textual retention is reported separately.
- **Missing evidence:** unavailable or ambiguous evidence cannot distinguish stages. Empty output is not automatically missing evidence.

The analyzer preserves the team's `api_failure` flag, but labels the aggregate **API or pipeline failures** because the runner catches general exceptions, not only network errors. `calls.parse_status` is written before parsing and cannot establish valid model JSON; use saved `runs.parse_events` and raw replies instead.

## Upstream schema and configuration mismatches

A wrong PLAN tool is an upstream warning, not proof that a fact was destroyed. For example, `schedule_transaction` has no `id` slot, but `EVIDENCE: scheduled id=7` can still reach Agent 2, which may choose `update_scheduled_transaction`. The existing `failure_stage.check` result is therefore retained as a PLAN/schema flag and **never used to remove a primary outcome or automatically label compression loss**. Section-style PLANs and multiple `call ... THEN call ...` clauses receive parser-uncertainty warnings because the existing parser does not resolve every such clause.

The current benchmark also has required earlier write tools absent from `allowed_tools` for `ws_13`, `ws_19`, `sl_11`, `sl_16`, `sl_18`, `sl_19`, `sl_20`, and `bk_15`. Agent 2 holds the tools, while destination/allowlist compressor prompts receive the narrower list. This may influence a compression comparison. The analysis records configuration-coverage warnings; it does not repair the task definitions or subtract affected tasks. These definitions require a separate team decision. All current critical reference keys have slots in their own required tool schemas.

## Estimates and interpretation

The report preserves each saved task-success bit, required-call hit count, and denominator. It reports observed task success, equal-task success after averaging available repeats, pooled required-call completion, macro task call completion, repeat variation, and existing suite/exposure strata. These aggregates can differ when trials are missing; the coverage table shows why. No new primary cohort is selected based on success, schema flags, or diagnostic labels.

Condition-minus-baseline comparisons match common task/repeat slots and report unmatched coverage plus per-task regressions and improvements. Confidence intervals resample whole tasks with their repeats, using a recorded random seed. Fewer than two tasks have no interval. This avoids treating several calls or repeats from the same task as independent benchmark items. Intervals are exploratory and may be degenerate for uniform outcomes.

The new runner uses independent upstream draws across conditions. Matching repeat indices does **not** create shared Agent 1 samples or isolate a causal compressor effect. Stage observations can support a mechanism hypothesis; they do not prove semantic sufficiency, instruction understanding, or mediation. Literal absence can reflect paraphrase, arithmetic, or an accepted alias; presence can be incidental, negated, or attached to a distractor. Such cases require review of the original traces. Reference coverage also excludes free prose that the team's scorer deliberately leaves unchecked.

Recorded prompt/output token counts are local text measurements or character estimates, not provider-billed usage. They are summed once from unique call records, including orphan groups, with missing-count coverage. Measured and estimated methods are not pooled. The nominal budget is not a hard token ceiling. No new token counting or vocabulary download is performed by the analyzer.

Some upstream metadata fields describe intent rather than independently verified execution: the configuration does not identify the mock backend, its control flag remains false although controls are planted in the current task documents, and several mode/seed fields have no corresponding runner switches. This analysis does not infer that those settings were experimentally varied. It preserves the configuration and actual trace evidence separately. Dirty source trees remain a provenance limitation even when the historical task blob matches.

## Reproduction

Run from this branch using the standard library; the analysis imports no external plotting or API client dependencies. Existing optional measured-token tests can skip when the tokenizer is unavailable.

```sh
PYTHONDONTWRITEBYTECODE=1 MAS_BACKEND=mock python3 run_tests.py

# Existing experiment directories, one per condition, with compatible configs.
# These commands only read saved outputs; they do not run models.
python3 analyze_redesign.py \
  results/redesign/BASELINE_RUN_ID results/redesign/COMPRESSION_RUN_ID \
  --tasks tasks/sample_tasks_63.json \
  --out-dir /tmp/redesign-analysis \
  --bootstrap-draws 2000 --seed 1729
```

Use a fresh empty output directory. `--fixture` explicitly labels mock or deterministic input as **non-empirical**. The upstream configuration does not record whether `--mock` was used, so the default label is `backend_unverified_from_upstream_metadata`; a model name in `config.json` is not proof of a real API run. Preserve external execution evidence before making empirical claims.

Outputs are `analysis.json` (estimates, paired differences, coverage, configurations, input and analysis-code SHA256s, Python version and analysis settings), `trials.csv` (unaltered saved scores and metrics), `diagnostics.jsonl` (field evidence and separate warnings), and `report.md`. The source hashes allow input or implementation changes to be detected. A missing historical dataset blob or dirty executed tree is explicitly unverified rather than silently certified. Infinite upstream CER-stratum bounds are serialized as strings; the original configuration bytes remain identified by their hash.

No paid experiment results accompany this contribution. Offline fixtures validate bookkeeping and attribution boundaries; they do not establish a compression benefit, harm, or statistical significance for the paper. Existing traces, scorer, task files, inference settings, and experiment entry points are unchanged.
