"""Eval harness for the Repo Interview Coach — the analysis half.

Two tiers, one scorecard:

  GATE  (deterministic, pure logic — no network, no model):
        guardrail arg-cleaning/clamping/blocking + Pydantic schema contracts.
        100% reproducible. This tier drives the exit code, so it is safe to run
        in CI and pre-commit.

  LIVE  (integration — real GitHub API + the configured model):
        the fetch tools' real-data and graceful-failure paths, and the
        load -> structure agent producing a grounded, typed ProjectProfile.
        Network/model checks tolerate a GitHub rate-limit as SKIP (an
        environment issue, not an agent bug), and the model check is graded
        pass@k because a small local model is non-deterministic. A hard FAIL
        here (agent genuinely misbehaved) still fails the run.

Each check is tagged [normal] / [edge] / [failure] so the coverage mix is
visible. Prints a PASS/FAIL/SKIP table and writes reproducible evidence
(`last_scorecard.txt`, `scorecard.html`). Exit is non-zero on any hard FAIL.

Locally the agent runs on Ollama (RIC_MODEL=ollama_chat/llama3.2) — no API
limits. Export GITHUB_TOKEN (e.g. `GITHUB_TOKEN=$(gh auth token)`) to lift the
GitHub rate limit from 60 to 5000 req/hr and keep the LIVE tier green.

Run:  uv run python -m evals.eval_harness   (from the project root)
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from google.genai import types

# Resolve paths relative to this file so the suite runs from anywhere — this
# folder lives inside the package (repo_interview_coach/evals/). Put the
# package's PARENT on sys.path so `import repo_interview_coach` resolves, and
# load the package-local .env by absolute path.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../repo_interview_coach
_PARENT = os.path.dirname(_PKG_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
load_dotenv(os.path.join(_PKG_DIR, ".env"))

from google.adk.runners import InMemoryRunner

from repo_interview_coach.agent import _BATCH_SIZE, _MAX_ROUNDS, _interview_plan, _wants_more
from repo_interview_coach.analysis_app.agent import root_agent as analysis_agent
from repo_interview_coach.github_tools import (
    fetch_file_tree,
    fetch_languages,
    fetch_recent_commits,
    fetch_repo_overview,
)
from repo_interview_coach.guardrails import sanitize_github_args
from repo_interview_coach.schemas import AnswerScore, ProjectProfile

OWNER, REPO = "Akshu24Tech", "repo-guardian"
EXPECTED_TOOLS = {
    "fetch_repo_overview", "fetch_readme", "fetch_recent_commits",
    "fetch_file_tree", "fetch_languages",
}
GHOST_REPO = ("Akshu24Tech", "this-repo-does-not-exist-xyz-000")
ANALYSIS_ATTEMPTS = 3  # pass@k: a small local model is non-deterministic.

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class _Tool:
    name = "fetch_readme"


def _rate_limited(x) -> bool:
    """True if a tool result is a GitHub rate-limit error (env issue, not a bug)."""
    s = str(x).lower()
    return "rate limit" in s


async def _run_analysis(owner_repo: str):
    runner = InMemoryRunner(agent=analysis_agent, app_name="eval")
    sess = await runner.session_service.create_session(app_name="eval", user_id="e")
    msg = types.Content(role="user", parts=[types.Part(text=owner_repo)])
    tool_calls: list[str] = []
    async for ev in runner.run_async(user_id="e", session_id=sess.id, new_message=msg):
        for fc in ev.get_function_calls():
            tool_calls.append(fc.name)
    final = await runner.session_service.get_session(app_name="eval", user_id="e", session_id=sess.id)
    return tool_calls, final.state.get("profile")


def _as_profile(raw) -> ProjectProfile | None:
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            return ProjectProfile.model_validate_json(raw)
        return ProjectProfile.model_validate(raw)
    except Exception:
        return None


def _lang(s: str) -> str:
    """Normalise a reported language: the small model sometimes echoes the raw
    'Language: 45945' (name + byte count) from tool output, so keep the name."""
    return str(s).split(":")[0].strip().lower()


async def main() -> int:
    results: list[tuple[str, str, str, str, str]] = []  # tier, name, kind, outcome, detail

    def check(tier: str, name: str, kind: str, ok, detail: str = "") -> None:
        outcome = ok if ok in (PASS, FAIL, SKIP) else (PASS if ok else FAIL)
        results.append((tier, name, kind, outcome, detail))

    def raises(fn) -> bool:
        try:
            fn()
            return False
        except Exception:
            return True

    t = _Tool()

    # ===================================================================== #
    #  GATE — guardrail. Pure logic, no network. Clean / clamp / block args. #
    # ===================================================================== #

    # [normal] a pasted full GitHub URL is normalised to owner/repo.
    a = {"owner": "https://github.com/Akshu24Tech/repo-guardian", "repo": ""}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.parses_url", "normal",
          a.get("owner") == OWNER and a.get("repo") == REPO, str(a))

    # [normal] "owner/repo" pasted into the single owner field is split.
    a = {"owner": "Akshu24Tech/repo-guardian", "repo": ""}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.parses_owner_slash_repo", "normal",
          a.get("owner") == OWNER and a.get("repo") == REPO, str(a))

    # [edge] a trailing ".git" (copied clone URL) is stripped from the repo name.
    a = {"owner": "Akshu24Tech", "repo": "repo-guardian.git"}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.strips_git_suffix", "edge", a.get("repo") == REPO, str(a.get("repo")))

    # [edge] an absurd max_files is clamped to the safe ceiling (200).
    a = {"owner": OWNER, "repo": REPO, "max_files": 99999}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.clamps_max_files", "edge", a.get("max_files") == 200, str(a.get("max_files")))

    # [edge] an absurd limit is clamped to its own ceiling (30).
    a = {"owner": OWNER, "repo": REPO, "limit": 99999}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.clamps_limit", "edge", a.get("limit") == 30, str(a.get("limit")))

    # [edge] a max_chars below the floor is raised to the minimum (500).
    a = {"owner": OWNER, "repo": REPO, "max_chars": 1}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.clamps_max_chars_floor", "edge", a.get("max_chars") == 500, str(a.get("max_chars")))

    # [edge] a non-numeric numeric arg doesn't crash — it defaults to the ceiling.
    a = {"owner": OWNER, "repo": REPO, "max_files": "lots"}
    sanitize_github_args(t, a, None)
    check("gate", "guardrail.non_numeric_arg_defaults", "edge", a.get("max_files") == 200, str(a.get("max_files")))

    # [failure] path-traversal / shell metacharacters -> tool never fires.
    blocked = sanitize_github_args(t, {"owner": "../../etc", "repo": "$(rm -rf)"}, None)
    check("gate", "guardrail.blocks_malicious", "failure",
          isinstance(blocked, dict) and "error" in blocked, str(blocked))

    # [failure] empty owner/repo -> blocked with a clean error, not a bad API call.
    blocked = sanitize_github_args(t, {"owner": "", "repo": ""}, None)
    check("gate", "guardrail.blocks_empty", "failure",
          isinstance(blocked, dict) and "error" in blocked, str(blocked))

    # ===================================================================== #
    #  GATE — schema. Pure logic. The typed contracts reject bad data.       #
    # ===================================================================== #

    # [failure] a profile missing required fields is rejected.
    check("gate", "schema.rejects_incomplete_profile", "failure",
          raises(lambda: ProjectProfile.model_validate({"owner": "x"})))

    # [failure] an out-of-range answer score (ge=1, le=5) is rejected.
    check("gate", "schema.rejects_out_of_range_score", "failure",
          raises(lambda: AnswerScore(question="q", score=9, strengths="s", gaps="g", model_answer="m")))

    # [normal] a complete, valid profile validates and round-trips cleanly.
    good = {
        "owner": OWNER, "repo": REPO, "summary": "A guardrail agent.",
        "primary_language": "Python", "languages": ["Python"],
        "notable_components": ["cli"], "recent_work": ["init"], "likely_weak_spots": ["tests"],
    }
    check("gate", "schema.valid_profile_roundtrips", "normal",
          not raises(lambda: ProjectProfile.model_validate(good).model_dump()))

    # ===================================================================== #
    #  GATE — interview state machine. Pure logic driving the batched HITL.  #
    #  Q = a question answer, "yes"/"no" = a round-gate answer.              #
    # ===================================================================== #

    Q = "answer"
    def act(responses):  # convenience: the action string for a response sequence
        return _interview_plan(responses)["action"]

    # [normal] empty history -> ask the very first question (k=0).
    plan0 = _interview_plan([])
    check("gate", "interview.starts_with_question", "normal",
          plan0["action"] == "question" and plan0["k"] == 0, str(plan0))

    # [normal] after a full batch of answers -> show the round review + gate.
    plan_g = _interview_plan([Q, Q, Q])
    check("gate", "interview.gate_after_batch", "normal",
          plan_g["action"] == "gate" and plan_g["round"] == 1, str(plan_g))

    # [normal] "yes" at the gate -> continue into the next round's first question.
    plan_y = _interview_plan([Q, Q, Q, "yes"])
    check("gate", "interview.yes_continues", "normal",
          plan_y["action"] == "question" and plan_y["k"] == _BATCH_SIZE, str(plan_y))

    # [failure] "no" at the gate -> stop (this is the bug the user hit: no more looping).
    check("gate", "interview.no_stops", "failure",
          _interview_plan([Q, Q, Q, "no"]) == {"action": "stop", "reason": "user"})

    # [edge] "no more" -> negation wins over the word "more" -> stop.
    check("gate", "interview.no_more_stops", "edge",
          act([Q, Q, Q, "no more"]) == "stop")

    # [edge] an ambiguous gate answer defaults to STOP, never an accidental loop.
    check("gate", "interview.ambiguous_stops", "edge",
          act([Q, Q, Q, "hmm maybe"]) == "stop")

    # [normal] two clean rounds then "no" -> stop after 6 questions.
    two_rounds = [Q, Q, Q, "yes", Q, Q, Q, "no"]
    check("gate", "interview.two_rounds_then_stop", "normal",
          act(two_rounds) == "stop", str(_interview_plan(two_rounds)))

    # [failure] hard cap: even with endless "yes", it stops at _MAX_ROUNDS.
    capped = ([Q, Q, Q, "yes"] * _MAX_ROUNDS)
    check("gate", "interview.hard_cap_stops", "failure",
          _interview_plan(capped) == {"action": "stop", "reason": "cap"}, str(_interview_plan(capped)))

    # [edge] gate-word interpreter: negation beats affirmation, unknown -> stop.
    check("gate", "interview.wants_more_semantics", "edge",
          _wants_more("yes") and not _wants_more("no")
          and not _wants_more("no more") and not _wants_more("") and _wants_more("sure keep going"))

    # ===================================================================== #
    #  LIVE — tools. Real GitHub API. Real data + graceful failure paths.    #
    #  SKIP (not FAIL) when GitHub rate-limits us — an environment issue.     #
    # ===================================================================== #

    ov = fetch_repo_overview(OWNER, REPO)
    # [normal] overview returns real, correctly-typed data for a live repo.
    check("live", "tools.overview_real_data", "normal",
          SKIP if _rate_limited(ov) else ("error" not in ov and ov.get("primary_language") == "Python"),
          str(ov)[:80])

    commits = fetch_recent_commits(OWNER, REPO, limit=3)
    # [normal] recent commits come back as dated "YYYY-MM-DD: msg" strings.
    check("live", "tools.commits_are_dated", "normal",
          SKIP if _rate_limited(commits) else (
              isinstance(commits, list) and len(commits) >= 1
              and commits[0][:2].isdigit() and ": " in commits[0]),
          str(commits[:1]))

    # [failure] a non-existent repo returns a graceful error dict, never raises.
    ghost = fetch_repo_overview(*GHOST_REPO)
    check("live", "tools.missing_repo_is_error", "failure",
          SKIP if _rate_limited(ghost) else (isinstance(ghost, dict) and "error" in ghost), str(ghost))

    # [failure] file_tree propagates the not-found error instead of crashing.
    tree = fetch_file_tree(*GHOST_REPO)
    check("live", "tools.file_tree_propagates_error", "failure",
          SKIP if _rate_limited(tree) else (
              isinstance(tree, list) and len(tree) == 1 and "ERROR" in tree[0]),
          str(tree)[:80])

    # ===================================================================== #
    #  LIVE — analysis agent. Local model + network. Graded pass@k because   #
    #  a small local model is non-deterministic (see DEBUG-LOG).             #
    # ===================================================================== #

    real_langs_raw = fetch_languages(OWNER, REPO)
    gh_down = _rate_limited(real_langs_raw) or "error" in (real_langs_raw or {})
    real_langs = set() if gh_down else {_lang(k) for k in real_langs_raw}

    best = {"fires": False, "valid": False, "identity": False, "langs": False}
    attempts_used = 0
    for attempt in range(1, ANALYSIS_ATTEMPTS + 1):
        attempts_used = attempt
        try:
            tool_calls, profile_raw = await _run_analysis(f"{OWNER}/{REPO}")
        except Exception as exc:  # a transient model/runtime error — try again
            best.setdefault("err", str(exc)[:80])
            continue
        prof = _as_profile(profile_raw)
        reported = {_lang(x) for x in ((prof.languages or [prof.primary_language]) if prof else []) if str(x).strip()}
        best["fires"] |= EXPECTED_TOOLS.issubset(set(tool_calls))
        best["valid"] |= prof is not None
        best["identity"] |= bool(prof) and prof.owner == OWNER and prof.repo == REPO
        best["langs"] |= bool(reported) and (gh_down or reported.issubset(real_langs))
        if all((best["fires"], best["valid"], best["identity"], best["langs"])):
            break

    ax = f"(pass@{attempts_used})"
    # [normal] across up to k tries, the agent fires all five GitHub tools.
    check("live", "analysis.fires_expected_tools", "normal", best["fires"], ax)
    # [normal] it produces output that parses into the typed ProjectProfile.
    check("live", "analysis.produces_valid_profile", "normal", best["valid"], ax)
    # [normal] identity is grounded: owner/repo come straight from the input.
    check("live", "analysis.identity_grounded", "normal", best["identity"], ax)
    # [normal] languages are grounded, not hallucinated — every reported language
    # really exists in the repo (SKIP if GitHub was down so we can't verify).
    check("live", "analysis.languages_not_hallucinated", "normal",
          SKIP if gh_down else best["langs"], ax)

    return _report(results)


def _report(results) -> int:
    order = {"gate": 0, "live": 1}
    results = sorted(results, key=lambda r: order[r[0]])
    p = sum(o == PASS for *_, o, _ in results)
    f = sum(o == FAIL for *_, o, _ in results)
    s = sum(o == SKIP for *_, o, _ in results)
    total = len(results)

    lines = ["=== Repo Interview Coach — eval scorecard ==="]
    cur = None
    for tier, name, kind, outcome, detail in results:
        if tier != cur:
            cur = tier
            label = "GATE (deterministic — drives exit code)" if tier == "gate" else "LIVE (real GitHub API + model)"
            lines.append(f"\n-- {label} --")
        note = f"  ({detail})" if (detail and outcome != PASS) else ""
        lines.append(f"[{outcome}] {kind:<7} {name}{note}")
    lines.append(f"\n--- {p} passed · {f} failed · {s} skipped  (of {total}) ---")

    print("\n" + "\n".join(lines))
    _write_artifacts(results, p, f, s, total)
    return 0 if f == 0 else 1


def _write_artifacts(results, p, f, s, total) -> None:
    """Write reproducible evidence next to this harness: a plain-text log and a
    self-contained HTML render (screenshot it for the visual pass/fail proof).
    Regenerated every run, so the evidence can never drift from the code."""
    here = os.path.dirname(__file__)
    summary = f"{p} passed · {f} failed · {s} skipped (of {total})"

    txt = ["=== Repo Interview Coach — eval scorecard ==="]
    cur = None
    for tier, name, kind, outcome, _ in results:
        if tier != cur:
            cur = tier
            txt.append(f"\n-- {'GATE (deterministic)' if tier == 'gate' else 'LIVE (GitHub API + model)'} --")
        txt.append(f"[{outcome}] {kind:<7} {name}")
    txt.append(f"\n--- {summary} ---")
    with open(os.path.join(here, "last_scorecard.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")

    def row(tier, name, kind, outcome):
        oc = outcome.lower()
        return (f'<tr><td class="b {oc}">{outcome}</td>'
                f'<td class="k {kind}">{kind}</td><td class="n">{name}</td></tr>')
    body, cur = [], None
    for tier, name, kind, outcome, _ in results:
        if tier != cur:
            cur = tier
            label = "GATE — deterministic (drives exit code)" if tier == "gate" else "LIVE — real GitHub API + model"
            body.append(f'<tr class="sec"><td colspan="3">{label}</td></tr>')
        body.append(row(tier, name, kind, outcome))
    rows = "\n".join(body)
    foot_cls = "ok" if f == 0 else "bad"
    html = f"""<!doctype html><meta charset="utf-8">
<title>Repo Interview Coach — eval scorecard</title>
<style>
  body{{background:#0d1117;color:#c9d1d9;font:14px/1.6 'Cascadia Code',Consolas,monospace;margin:0;padding:32px}}
  .card{{max-width:780px;margin:auto;background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden}}
  h1{{font-size:16px;margin:0;padding:16px 20px;border-bottom:1px solid #30363d;color:#e6edf3}}
  h1 small{{color:#8b949e;font-weight:400}}
  table{{width:100%;border-collapse:collapse}}
  td{{padding:6px 12px;border-bottom:1px solid #21262d}}
  tr.sec td{{background:#0d1117;color:#8b949e;font-weight:700;font-size:12px;letter-spacing:.4px;text-transform:uppercase;border-bottom:1px solid #30363d}}
  .b{{font-weight:700;width:56px}} .pass{{color:#3fb950}} .fail{{color:#f85149}} .skip{{color:#8b949e}}
  .k{{width:80px;text-transform:uppercase;font-size:11px;letter-spacing:.5px}}
  .normal{{color:#58a6ff}} .edge{{color:#d29922}} .failure{{color:#bc8cff}}
  .n{{color:#c9d1d9}}
  .foot{{padding:14px 20px;font-size:15px;font-weight:700}}
  .ok{{color:#3fb950}} .bad{{color:#f85149}}
</style>
<div class="card">
  <h1>Repo Interview Coach — eval scorecard&nbsp; <small>uv run python -m evals.eval_harness</small></h1>
  <table>{rows}</table>
  <div class="foot {foot_cls}">{summary}</div>
</div>
"""
    with open(os.path.join(here, "scorecard.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
