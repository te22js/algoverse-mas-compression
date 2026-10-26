# Compression redesign — implementation plan

Written after inspecting the repository, before any implementation. Phase 1
has not started.

- snapshot tag: `pre-compression-redesign` → `7608c47`
- archive: `archive/pre_compression_redesign/` (+ README recording config,
  baseline, known problems, restore instructions)
- working branch: `compression-redesign`, forked from `9cb4883`
- previous branch `multi-suite-and-plan-regime` left untouched at `9cb4883`

---

## 1. What is actually here

### Entry points

There are **two runners** and the distinction matters.

| file | role |
|---|---|
| `run.py` | the team's original runner. Sweeps compression conditions. Not what produced any current result. |
| `baseline.py` | **the real entry point.** 40 KB. Run 1 only, condition hard-coded to `"none"`. |

`baseline.py` is not a script that uses `pipeline.py` — it is a **monkey-patch
layer over it**. At import it reassigns seven module attributes:

```
pipeline.TOOL_REGISTRY        = environment.TOOL_REGISTRY     (line  88)
pipeline.score_run            = score_checkable               (line 233)
pipeline._build_agent1_input  = build_without_markers         (line 266)
pipeline._parse               = _parse_tolerant               (line 353)
pipeline.run_pipeline         = run_with_schema               (line 567)
pipeline._replay              = _replay_safe                  (line 588)
pipeline.run_pipeline         = _run_capturing                (line 938)
```

Note `run_pipeline` is replaced **twice** — `_run_capturing` wraps
`run_with_schema` wraps the original. Any change to the pipeline contract has
to respect that chain, and the order of those assignments is load-bearing.

### Module map against Part 0.1

| requested | file | symbol |
|---|---|---|
| pipeline | `pipeline.py` | `run_pipeline`, `_build_agent1_input`, `_replay`, `_parse` |
| Agent 1 prompt | `agent1_prompts.py` | `REGIMES` dict: summarise / plan / verbatim / scoped / defended; `brief` lives in `baseline.py` as `_BRIEF_SYSTEM` |
| compressor | `compressors.py` | `compress()`, `BUDGET_TOKENS`, six conditions |
| Agent 2 prompt | `baseline.py` | `_AGENT2_TEMPLATE`, `_PerThreadSystem`, `_signatures`, `TOOL_PARAMS`, `TOOL_DESC` |
| task loading / tiering | `benign_all.py` | `build_suite`, `tier_of`, `THIN_BELOW`, `OVERSIZED_ABOVE` |
| scoring | `baseline.py` | `score_checkable`, `_args_match`, `_norm` |
| benign critical fields | `obtainable.py` | `prune`, `findable`, `PROSE_KEYS`, `DATE_KEYS` |
| adversarial scoring | `baseline.py` | inside `score_checkable`; `harmful_critical` is **null on all 51 tasks** |
| synthetic expansion | `filler.py`, `filler_suites.py`, `suites.py` | `inflate_*`, `Gen`, `transaction`, `unit_*` |
| control markers | `to_harness_all.py` | `C_TEXT`, `N1_TEXT`, `N2_TEXT` — **fields only, never planted** |
| raw capture | `baseline.py` | patch 11, `_run_capturing`, `_cap` thread-local |
| budgets | `compressors.py` | `BUDGET_TOKENS` (env `MAS_BUDGET_TOKENS`), default 150 |
| tests | — | **none. Zero test functions in the repository.** |
| outputs | `baseline_results.json`, `run1_log.txt` | archived |

### Where the compressor sits

`pipeline.py:77` — one call:

```python
compressed = compress(condition, handoff, compressor_backend, task)
```

**Nothing in `pipeline.py` or `compressors.py` counts a token.** Every metric
in Parts 3, 19 and 20 has to be added; none can be derived from stored data.

---

## 2. Conflicts between the brief and the code

These are Part 31 stop conditions. Three need a decision from you; the rest I
can work around.

### 2.1 Competing implementations — Part 31 item 6 — **needs a decision**

The repository contains duplicated pipelines:

| | |
|---|---|
| `to_harness.py` vs `scripts/to_harness.py` | **different files**, same name, both build task JSON |
| `benign.py` vs `benign_all.py` | workspace-only vs all four suites |
| `tasks/sample_tasks.json` vs `tasks/sample_tasks_all.json` | 18-task legacy vs 51-task current |
| `run.py` vs `baseline.py` | two runners |

`run1_log.txt` came from `baseline.py` + `tasks/sample_tasks_all.json`. I know
that because I wrote the log generator. A reader of the repo cannot tell.

**Proposal:** move the legacy workspace-only path to `legacy/` with a README
rather than deleting it, and leave `run.py` alone. **Not doing this until you
say so** — Part 32 says do not delete the old experiment, and I would rather
confirm than guess which half you consider the old experiment.

### 2.2 Agent 2 does not see the task — Part 31 item 2 — **needs a decision**

Verified in `pipeline.py:84`:

```python
a2_input = f"Handoff from Agent 1:\n{compressed}"
```

That is the whole input. Agent 2 receives no task text, which is why
`obtainable.prune` forces the PLAN to carry task-stated values: measured, **40
of 117 required values exist only in the task prompt**.

Part 8 wants Agent 2 to do the reasoning — infer "March amount = 50, so
difference = 5". But Agent 2 cannot know the task says *"pay the March
difference"* unless something tells it. Three options:

| option | effect on the instrument |
|---|---|
| A. Agent 1 writes a one-line objective into the handoff | objective is compressible, so it can be lost — consistent with the current design |
| B. Agent 2 is given the task text directly | Agent 2 can recover from a lossy handoff; compression damage becomes invisible. **Breaks the instrument.** |
| C. Agent 2 gets a fixed abstract goal, no task specifics | weakest; most tasks become unanswerable |

**I recommend A** and will not implement B without you saying so explicitly,
because it silently removes the asymmetry the whole study rests on.

### 2.3 `harmful_critical` cannot be populated yet — Part 31 item 5 — **blocked**

`harmful_critical` is null on all 51 tasks because **no attack specs exist
yet**. `tasks/attack_tasks.json` is 3.6 KB of the team's original mock
scaffolding, not AgentDojo injections. Sourayan owns the attack layer and has
not delivered.

Part 12 asks for symmetric critical-field scoring. I can build the
**mechanism** and test it against synthetic attack specs. I cannot populate
`harmful_critical` for real attacks that do not exist. Flagged rather than
invented, per Part 32.

### 2.4 Not blocking, but worth stating

- **No tests exist.** Part 33's test suite starts from zero, not from
  extending something.
- **`baseline.py` is a patch layer, not a program.** Part 28's centralised
  config is a genuine refactor of 40 KB of monkey-patching, not a config file
  drop-in. I propose introducing config alongside the patches and migrating
  incrementally rather than rewriting.
- **`tier_of` already filters tasks** on raw context size (`THIN_BELOW=1000`).
  That is a second, older eligibility mechanism that predates CER and will
  interact with it. Parts 4 and 24 need to know it exists.

---

## 3. Where each requested change goes

### Part 3 — CER and token accounting

| | |
|---|---|
| files | new `metrics.py`; `pipeline.py`; `baseline.py` |
| change | `count_tokens()` using tiktoken with a chars/4 fallback. `run_pipeline` records `source_context_tokens`, `precompression_handoff_tokens`, `compressor_budget`, `compressed_tokens`, `cer`, `realized_ratio`, `fraction_removed` |
| old behaviour | preserved — additive fields only |
| risk | tiktoken needs a network fetch for its vocab; the fallback must be explicit in the record so nobody quotes an estimate as measured |
| validation | unit tests on known strings; zero/empty handoff must not divide by zero |

### Part 4 — strata

| | |
|---|---|
| files | new `config.py`; `metrics.py` |
| change | `CER_STRATA = {"unexposed": (0,1.0), "weak": (1.0,1.5), "moderate": (1.5,3.0), "strong": (3.0,inf)}`, `PRIMARY_CER_MIN = 1.5` |
| note | strata are an **analysis label**, never a filter in the runner. Part 24. |

### Part 10/11 — controls planted and tracked per stage

| | |
|---|---|
| files | `benign_all.py` (planting), new `controls.py` (detection), `pipeline.py` (capture) |
| change | insert C/N1/N2 into the documents each task actually reads, reusing `filler.block_boundaries`; record `present_in_documents / handoff / compressed / expressed_in_action` |
| risk | **a planted control must not change a ground-truth answer.** `verify_all.py` already catches that and must pass after planting |
| validation | assert each control string appears in `documents` for every task; `verify_all` still 57 pass / 0 broken |

### Part 12 — symmetric scoring

| | |
|---|---|
| files | `baseline.py` (`score_checkable`), `obtainable.py` |
| change | `harmful_success` matches `harmful_critical` the way benign matches `handoff_critical`; prose excluded unless an attack declares it critical |
| blocked on | §2.3 |

### Part 13 — failure taxonomy

| | |
|---|---|
| files | new `failure_stage.py` |
| change | classify from raw traces only; `unknown` when evidence is absent |
| key rule | if the correct record is in the handoff and the PLAN already names a distractor, that is `AGENT1_SELECTION`, **never** `COMPRESSOR_LOSS` (Part 13, bk_5) |

### Part 15 — shared mutable state

Audit beyond the one known bug. Current per-task writes to module globals:

- `pipeline.AGENT2_SYSTEM = _PerThreadSystem()` — written per task inside
  `run_with_schema`. Reads thread-local, so it is safe, but it is still a
  global write in a concurrent path and should become explicit passing.
- `_a2.prompt`, `_cap.buf` — thread-locals, correct, but untested.

Proposed: a `TaskRunContext` passed explicitly; globals kept as a shim so the
old path still runs. Add a concurrency test asserting serial and 8-worker runs
produce identical per-task configuration.

### Part 7/8 — evidence regime

| | |
|---|---|
| files | `agent1_prompts.py` (new `EVIDENCE` entry), `baseline.py` |
| change | `handoff_regime = plan | evidence` via existing `MAS_AGENT1` |
| **`plan` is not removed** | Part 32, and old results stay reproducible |
| depends on | §2.2 decision |

### Part 6/18 — native vs stress_test

| | |
|---|---|
| files | `config.py`, `benign_all.py`, `suites.py` |
| change | `experiment_mode`; native disables `INFLATE`; stress_test keeps it with levels. Record `native_tokens`, `synthetic_tokens`, `synthetic_fraction` per task |
| note | **native will produce mostly unexposed tasks** — slack is 87 native tokens median. That is the finding, not a bug (Part 26) |

### Part 17 — budget sweep

`compressors.BUDGET_TOKENS` becomes a parameter rather than a module global
read at import. Today it is read once, so a sweep in one process is impossible.

### Part 20/21 — analysis exports

New `export_runs.py` → JSONL + CSV with the Part 20 columns. No plots until
the columns exist.

---

## 4. Phase 1, which is what I propose to do next

Per Part 30, and nothing beyond it.

1. `config.py` — `RunConfig` dataclass, saved with every run
2. `metrics.py` — token counting, CER, realized ratio, fraction removed
3. `pipeline.py` — record sizes at every stage; **`source_context_tokens` and
   `precompression_handoff_tokens` as separate fields** (Part 19)
4. strata labelling in `metrics.py`, thresholds in `config.py`
5. raw-trace capture hardened and written as JSONL to `results/redesign/<run_id>/`
6. API-failure handling audited against Part 14 — already excludes, needs
   per-call `requested/succeeded/error/retries/parse_status` records
7. shared-state audit + concurrency test
8. run-id, git commit, dirty flag, prompt hashes persisted

Then: unit tests, and a **4-task smoke run** — `bk_5` (unexposed), `ws_19`
(strong), `sl_15` (prose answer), `bk_9` (qualifier) — roughly 8 API calls.

**No full 51-task run.**

---

## 5. What I need from you before going further than Phase 1

1. **§2.2** — Agent 2 and the task. Option A, or something else?
2. **§2.1** — may I move the workspace-only legacy path to `legacy/`?
3. **§2.3** — confirm `harmful_critical` stays unpopulated until attacks exist,
   and that building the mechanism against synthetic specs is acceptable.

Phase 1 touches none of those, so I can start on it while you decide.
