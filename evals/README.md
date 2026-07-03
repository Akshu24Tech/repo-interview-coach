# Evals — Repo Interview Coach

This project ships a real evaluation suite for its analysis half. **20 cases**,
each tagged `[normal]` / `[edge]` / `[failure]`, split across two tiers.

Run from the package root (the folder containing `agent.py` and this `evals/`):

```bash
GITHUB_TOKEN=$(gh auth token) python -m evals.eval_harness
# --> 20 passed · 0 failed · 0 skipped
```

Every run writes reproducible evidence next to the harness — `last_scorecard.txt`
(committed) and `scorecard.html` (open it in a browser for the visual pass/fail
proof) — so the evidence can never drift from the code.

## Tier 1 — GATE (deterministic, drives the exit code)
Pure logic. No network, no model, so it is 100% reproducible and safe to gate CI
and pre-commit. Covers the security guardrail and the typed schema contracts.

| Check | Kind | What it proves |
|---|---|---|
| `guardrail.parses_url` | normal | a pasted GitHub URL → `owner/repo` |
| `guardrail.parses_owner_slash_repo` | normal | `owner/repo` in one field is split |
| `guardrail.strips_git_suffix` | edge | a trailing `.git` is removed |
| `guardrail.clamps_max_files` | edge | out-of-range `max_files` → 200 |
| `guardrail.clamps_limit` | edge | out-of-range `limit` → 30 |
| `guardrail.clamps_max_chars_floor` | edge | below-floor `max_chars` → 500 |
| `guardrail.non_numeric_arg_defaults` | edge | a non-numeric arg doesn't crash |
| `guardrail.blocks_malicious` | failure | `../../etc`, shell metachars → tool never fires |
| `guardrail.blocks_empty` | failure | empty owner/repo → clean error, no bad API call |
| `schema.rejects_incomplete_profile` | failure | a profile missing required fields is rejected |
| `schema.rejects_out_of_range_score` | failure | an answer score outside 1–5 is rejected |
| `schema.valid_profile_roundtrips` | normal | a valid `ProjectProfile` validates + round-trips |

## Tier 2 — LIVE (real GitHub API + the configured model)
Exercises the fetch tools and the `load → structure` agent end-to-end.

| Check | Kind | What it proves |
|---|---|---|
| `tools.overview_real_data` | normal | the GitHub tools hit the live API and return real data |
| `tools.commits_are_dated` | normal | commits come back as dated `YYYY-MM-DD: msg` strings |
| `tools.missing_repo_is_error` | failure | a non-existent repo → graceful error dict, never raises |
| `tools.file_tree_propagates_error` | failure | `file_tree` propagates not-found instead of crashing |
| `analysis.fires_expected_tools` | normal | the agent calls all 5 GitHub tools |
| `analysis.produces_valid_profile` | normal | output parses into the typed `ProjectProfile` |
| `analysis.identity_grounded` | normal | profile owner/repo match the real repo |
| `analysis.languages_not_hallucinated` | normal | every reported language actually exists in the repo |

Two robustness decisions make this tier reliable without weakening it:
- **Rate-limit → SKIP, not FAIL.** If GitHub rate-limits us (60 req/hr
  unauthenticated), the affected checks skip — that's an environment issue, not
  an agent bug. Export `GITHUB_TOKEN` (5000 req/hr) to keep them green.
- **The model check is graded pass@k.** The local model (llama3.2 via Ollama) is
  non-deterministic and a small model sometimes fires only one tool or drops a
  language. The agent gets up to 3 attempts and the check passes if it succeeds
  in any — a fair "the agent *can* do this" bar. A genuine misbehaviour across
  all attempts still fails the run.

> This harness has already earned its keep — it caught real issues: the
> `before_tool_callback` invoked by ADK with the keyword `tool_context=` (a
> positional unit test had missed it), and the small model echoing raw
> `Language: <bytecount>` strings into the profile (now normalised).

## The official ADK eval (the methodology) — `loader.evalset.json` + `test_config.json`
ADK-native evaluation via `AgentEvaluator`, run against the `analysis_app`
package. Scores tool trajectory + response match against thresholds in
`test_config.json`.

```bash
uv run adk eval analysis_app evals/loader.evalset.json
```

## Why only the analysis half is evaluated
The full coach includes a human-in-the-loop interview (`request_input`), which
pauses for the candidate and so cannot be scripted in a static evalset. The
analysis half is the deterministic, gradeable part — that is what these evals
target. The interview flow is verified interactively in `adk web`.

## Pytest gate
`test_evals.py` wraps both the deterministic harness (`main() == 0`) and the
official ADK evalset, so the suite runs under `pytest` in CI:

```bash
GITHUB_TOKEN=$(gh auth token) python -m pytest evals/test_evals.py
```
