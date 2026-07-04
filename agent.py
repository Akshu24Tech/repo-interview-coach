"""Repo Interview Coach — ADK 2.3 Workflow graph.

Graph:  START -> load_context -> structure_profile -> interview -> build_dossier

- load_context     : tool-calling LlmAgent, pulls the repo's real README,
                     commits, file tree, languages. Writes raw text to state.
- structure_profile: LlmAgent with output_schema=ProjectProfile (no tools, by
                     ADK rule), distils the raw context into a typed profile.
- interview        : LlmAgent with the built-in request_input HITL tool. Asks
                     the candidate one question at a time, waits for each answer
                     (human-in-the-loop pause), scores it, probes weak spots.
- build_dossier    : LlmAgent with output_schema=Dossier, turns the profile +
                     interview transcript into resume bullets, a question bank,
                     and an upgrade list.

State flows node-to-node via each agent's `output_key`, read downstream with
ADK `{state_key}` instruction templating.

Agents are built via factory functions so the same definitions can be reused
both in this Workflow and in the eval target (`eval_agent.py`) without an agent
instance ending up with two parents.
"""

from __future__ import annotations

import os
import re

from google.adk.agents import LlmAgent
from google.adk.tools._request_input_tool import request_input
from google.adk.workflow import START, Workflow, node
from google.genai import types as genai_types

from .github_tools import (
    fetch_file_tree,
    fetch_languages,
    fetch_readme,
    fetch_recent_commits,
    fetch_repo_overview,
)
from .guardrails import sanitize_github_args
from .schemas import Dossier, ProjectProfile, QuestionBank

# Model is env-configurable so every node can be routed without code edits.
# Default Gemini 2.5 Flash — its free-tier TPM comfortably handles both the
# context-heavy load/structure/dossier nodes and the multi-turn HITL interview.
# Set RIC_MODEL to a LiteLLM provider id (openrouter/…, anthropic/…, ollama/…)
# to route elsewhere.
MODEL_NAME = os.getenv("RIC_MODEL", "gemini-2.5-flash")
_LITELLM_PREFIXES = ("openai/", "openrouter/", "anthropic/", "ollama/", "ollama_chat/")


def _model(name: str = MODEL_NAME):
    """Return a model spec ADK understands: a bare Gemini id, or a LiteLlm
    wrapper for any provider-prefixed id (openrouter/…, anthropic/…, etc.)."""
    if name.startswith(_LITELLM_PREFIXES):
        from google.adk.models.lite_llm import LiteLlm

        # num_retries lets LiteLLM back off and honor Retry-After on rate limits.
        return LiteLlm(model=name, num_retries=5)
    return name

_LOAD_INSTRUCTION = """\
The user gives you a public GitHub project as "owner/repo" or a GitHub URL.
Parse the owner and repo. Call fetch_repo_overview first; if it errors (private
or not found), say so plainly and stop. Otherwise call fetch_readme,
fetch_recent_commits, fetch_file_tree and fetch_languages.

Then write a CONCISE plain-text briefing (aim for under ~250 words) grounded
ONLY in what the tools returned: what it does, its languages, the notable
technical components/modules visible in the file tree, what recent commits show
was built, and the weak spots a sharp interviewer would probe. Be dense, not
verbose — this briefing is re-read by later steps, so keep it tight. Never
invent anything."""

_STRUCTURE_INSTRUCTION = """\
Here is a briefing of a GitHub project:

{raw_context}

Produce a structured ProjectProfile from it. Use only facts present in the
briefing. If the project is thin, reflect that honestly in likely_weak_spots."""

_INTERVIEW_INSTRUCTION = """\
You are a sharp, demanding technical interviewer. The candidate built this project:

{profile}

You have a prepared bank of project-specific questions to draw from:

{question_bank}

Run the interview in ROUNDS of 3 questions. A [INTERVIEWER CONTROL] message tells
you each turn exactly what to do: ask one specific question (it names WHICH bank
question number to use), OR give a round review and ask whether to continue, OR
stop and write the transcript. Follow it EXACTLY. One request_input call per turn,
never batched. Base each question on the named bank item, but sharpen it for THIS
candidate and make it harder if the last answer was strong; probe a real weak
spot; never repeat an earlier question. If the control message points past the end
of the bank, ask a fresh, harder question on an area not yet probed. Be direct, no
flattery; if the candidate bluffs or hand-waves, call it out.

After each round of 3, the control message will tell you to give a short REVIEW (a
score out of 5 and a one-line strength/gap per question) and THEN ask, via
request_input, whether they want another round.

When the control message says the interview is over, STOP calling request_input and
output a plain-text transcript with one block PER ACTUAL INTERVIEW QUESTION asked
(exclude the yes/no "want another round" prompts; do not invent extra questions),
each in EXACTLY this format so the next step can parse it:

Q: <the question>
A: <the candidate's answer, summarised>
Score: <N>/5
Strength: <one line>
Gap: <one line — what was missing or wrong>
Model answer: <one tight line grounded in the project>

This transcript is for the next step."""

_QUESTIONBANK_INSTRUCTION = """\
Here is a structured profile of a candidate's project:

{profile}

Generate a QuestionBank: 8-10 tough, SPECIFIC technical interview questions for
THIS project, ordered from a warm-up to the hardest. Ground every question in the
profile — its real components, recent work, and likely weak spots. No generic
questions ("what is an API?"); each must only make sense for this exact project.
These drive a live interview and are reused in the final dossier."""

_DOSSIER_INSTRUCTION = """\
Project profile:

{profile}

Prepared question bank (reuse and refine these):

{question_bank}

Interview transcript:

{transcript}

Produce the final Dossier: 3-4 STAR-shaped resume bullets (honest, quantified
only where the project supports it); a question_bank that STARTS from the prepared
bank above (refine the wording and add any sharp questions the interview
surfaced); an upgrade list of concrete improvements that would both strengthen the
project and give better interview stories; and a short honest read on how
interview-ready it is."""


# The interview runs in ROUNDS of _BATCH_SIZE questions. After each round the
# coach shows a review and asks (human-in-the-loop) whether to continue; "no"
# stops and produces the dossier, "yes" starts another round — up to a hard cap
# of _MAX_ROUNDS so it can never loop forever. Progression is driven entirely by
# code counting answers in the session history, NOT by trusting the model to
# remember what it asked — a self-looping LlmAgent re-enters fresh on each
# request_input resume and weak models just re-ask the opener. See DEBUG-LOG
# 2026-06-30 / 07-02. The theme pool deepens across rounds.
_BATCH_SIZE = 3
_MAX_ROUNDS = 5  # hard safety cap: at most _MAX_ROUNDS * _BATCH_SIZE questions.
_INTERVIEW_THEMES = [
    "a 60-second pitch of what the project does and why it matters",
    "a key architecture or design decision, and the reasoning behind it",
    "the hardest bug, tradeoff, or failure they hit and how they handled it",
    "how the system handles failure: errors, retries, rate limits, bad input",
    "testing and how they know it actually works (evals, edge cases)",
    "security and trust boundaries: untrusted input, secrets, tool access",
    "scaling and cost: what breaks at 10x load or 10x data, and the API bill",
    "a concrete tradeoff they would revisit, and what they would do differently",
]
_REQUEST_INPUT_NAME = "adk_request_input"

# Words that decide the "want another round?" gate. Negation wins over "more"
# so answers like "no more" correctly STOP; anything unclear also stops, so the
# interview can never loop on garbage input.
_STOP_WORDS = {"no", "nope", "nah", "stop", "done", "enough", "quit", "finish", "end", "n"}
_GO_WORDS = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "continue", "more", "go", "keep", "another", "yea"}


def _wants_more(text: str) -> bool:
    """Interpret a gate answer. Negation is checked first (so 'no more' stops),
    then affirmation; anything ambiguous defaults to STOP so we never loop."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    if words & _STOP_WORDS:
        return False
    if words & _GO_WORDS:
        return True
    return False


def _interview_plan(responses: list[str]) -> dict:
    """Pure state machine for the batched interview. Given the ordered list of
    user answers to `request_input` (both interview answers AND the yes/no gate
    answers), decide the next move. Deterministic and side-effect free so it is
    unit-tested directly (see the eval harness). Returns one of:
      {"action": "question", "k": <0-based questions answered>}
      {"action": "gate", "round": <1-based round just finished>}
      {"action": "stop", "reason": "user" | "cap"}

    Layout: each round is _BATCH_SIZE questions followed by ONE gate answer, so
    within a cycle of length L = _BATCH_SIZE + 1 the response at cycle-position
    _BATCH_SIZE is a gate answer and the rest are interview answers."""
    n = len(responses)
    L = _BATCH_SIZE + 1
    gates_answered = sum(1 for i in range(n) if i % L == _BATCH_SIZE)
    questions_answered = n - gates_answered
    last_is_gate = n > 0 and (n - 1) % L == _BATCH_SIZE

    if last_is_gate:
        if gates_answered >= _MAX_ROUNDS:
            return {"action": "stop", "reason": "cap"}
        if not _wants_more(responses[-1]):
            return {"action": "stop", "reason": "user"}
        return {"action": "question", "k": questions_answered}

    if n % L == _BATCH_SIZE:
        return {"action": "gate", "round": questions_answered // _BATCH_SIZE}

    return {"action": "question", "k": questions_answered}


def _request_input_responses(callback_context, llm_request) -> list[str]:
    """Ordered list of the candidate's answers to `adk_request_input`, read from
    the authoritative session event history (falling back to per-turn contents if
    the session is absent — the Workflow resume rebuilds contents empty, which is
    why the old count stuck at 0; see DEBUG-LOG 2026-07-02)."""
    def _collect(iterable) -> list[str]:
        out: list[str] = []
        for item in iterable or []:
            content = getattr(item, "content", item)
            for part in getattr(content, "parts", None) or []:
                fr = getattr(part, "function_response", None)
                if fr is not None and getattr(fr, "name", None) == _REQUEST_INPUT_NAME:
                    out.append(str(getattr(fr, "response", "")))
        return out

    session = getattr(callback_context, "session", None)
    events = getattr(session, "events", None)
    if events:
        return _collect(events)
    return _collect(llm_request.contents)


def _strip_request_input(llm_request) -> None:
    """Physically remove the request_input tool so a weak model CANNOT keep
    asking — with no tool to call it must emit the final transcript. Instruction
    alone doesn't hold on small models (DEBUG-LOG: llama-3.3/scout looped)."""
    cfg = getattr(llm_request, "config", None)
    if cfg is not None and cfg.tools:
        for tool in cfg.tools:
            fds = getattr(tool, "function_declarations", None)
            if fds:
                tool.function_declarations = [
                    fd for fd in fds if fd.name != _REQUEST_INPUT_NAME
                ]
        cfg.tools = [t for t in cfg.tools if getattr(t, "function_declarations", None)]
    if getattr(llm_request, "tools_dict", None):
        llm_request.tools_dict.pop(_REQUEST_INPUT_NAME, None)


def _steer_interview(callback_context, llm_request):
    """before_model_callback: derive the next move from session history and inject
    a hard [INTERVIEWER CONTROL] directive (ask question N / review + gate / stop).
    Makes the batched progression deterministic on any model."""
    plan = _interview_plan(_request_input_responses(callback_context, llm_request))

    if plan["action"] == "stop":
        why = ("You have reached the maximum number of rounds."
               if plan.get("reason") == "cap"
               else "The candidate chose to stop.")
        directive = (
            f"The interview is over. {why} Do NOT call request_input again. Output the "
            "final plain-text transcript now — one block per ACTUAL interview question "
            "asked (exclude the yes/no 'want another round' prompts), in the exact format."
        )
        _strip_request_input(llm_request)
    elif plan["action"] == "gate":
        rnd = plan["round"]
        directive = (
            f"Round {rnd} complete — the candidate has answered {_BATCH_SIZE} questions this round. "
            f"FIRST output a brief REVIEW of just this round: for each of the last {_BATCH_SIZE} "
            "questions, one line 'Q<n>: <score>/5 — <strength> / <gap>'. Keep it tight. "
            "THEN call request_input asking EXACTLY: 'Want another round of "
            f"{_BATCH_SIZE} questions? Reply yes or no.' Do NOT ask a new interview question this "
            "turn — the only request_input is the yes/no."
        )
    else:  # question
        k = plan["k"]
        theme = _INTERVIEW_THEMES[k % len(_INTERVIEW_THEMES)]
        round_no = k // _BATCH_SIZE + 1
        in_round = k % _BATCH_SIZE + 1
        directive = (
            f"Interview in progress: {k} question(s) answered so far. Now ask question "
            f"{in_round} of {_BATCH_SIZE} in round {round_no}. Use bank question #{k + 1} as the basis "
            f"(if the bank has fewer than {k + 1} questions, ask a fresh harder question on an un-probed "
            f"area such as: {theme}). Sharpen it for this candidate and make it harder than the last; "
            "probe a real weak spot. Ask exactly ONE new question via request_input. Never repeat an "
            "earlier question. Ask ONLY the question."
        )

    llm_request.contents = list(llm_request.contents or []) + [
        genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="[INTERVIEWER CONTROL] " + directive)],
        )
    ]
    return None


def _fix_request_input_schema(callback_context, llm_response):
    """after_model_callback: weak models (small local Ollama ones especially)
    fill request_input's `response_schema` arg with a bare JSON-Schema type
    string like "string" instead of an object {"type": "string"}. ADK's HITL
    resume then does TypeAdapter("string") -> SchemaError. Coerce any non-object
    response_schema into {"type": "string"} so resume works on any model."""
    content = getattr(llm_response, "content", None)
    if content is None or not content.parts:
        return None
    changed = False
    for part in content.parts:
        fc = getattr(part, "function_call", None)
        if fc is None or fc.name != _REQUEST_INPUT_NAME or not fc.args:
            continue
        rs = fc.args.get("response_schema")
        if rs is not None and not isinstance(rs, dict):
            fc.args["response_schema"] = {"type": "string"}
            changed = True
    return llm_response if changed else None


def build_load_agent() -> LlmAgent:
    """Context Loader — calls the (guarded) GitHub tools, emits rich plain text."""
    return LlmAgent(
        name="load_context",
        model=_model(),
        description="Loads a public GitHub project's real content.",
        instruction=_LOAD_INSTRUCTION,
        tools=[
            fetch_repo_overview,
            fetch_readme,
            fetch_recent_commits,
            fetch_file_tree,
            fetch_languages,
        ],
        before_tool_callback=sanitize_github_args,
        output_key="raw_context",
    )


def build_structure_agent() -> LlmAgent:
    """Distils the raw briefing into a typed ProjectProfile (no tools allowed)."""
    return LlmAgent(
        name="structure_profile",
        model=_model(),
        description="Distils the raw project briefing into a typed ProjectProfile.",
        instruction=_STRUCTURE_INSTRUCTION,
        output_schema=ProjectProfile,
        output_key="profile",
    )


def build_questionbank_agent() -> LlmAgent:
    """Generates a project-specific bank of interview questions from the profile,
    BEFORE the interview — so follow-up rounds draw sharp, grounded questions
    instead of generic themes. Reused by the dossier. No tools (output_schema)."""
    return LlmAgent(
        name="question_bank",
        model=_model(),
        description="Generates a project-specific bank of interview questions from the profile.",
        instruction=_QUESTIONBANK_INSTRUCTION,
        output_schema=QuestionBank,
        output_key="question_bank",
    )


def build_interview_agent() -> LlmAgent:
    """Live, adaptive HITL interview via the request_input long-running tool."""
    return LlmAgent(
        name="interview",
        model=_model(),
        description="Conducts a live, adaptive technical interview about the project.",
        instruction=_INTERVIEW_INSTRUCTION,
        tools=[request_input],
        before_model_callback=_steer_interview,
        after_model_callback=_fix_request_input_schema,
        output_key="transcript",
    )


def build_dossier_agent() -> LlmAgent:
    """Builds the final typed Dossier."""
    return LlmAgent(
        name="build_dossier",
        model=_model(),
        description="Builds the final interview-prep dossier.",
        instruction=_DOSSIER_INSTRUCTION,
        output_schema=Dossier,
        output_key="dossier",
    )


def build_workflow() -> Workflow:
    """Wire the agents into the interview-coach graph:
    START -> load -> structure -> question_bank -> interview -> dossier."""
    load_node = node(build_load_agent(), name="load_context")
    structure_node = node(build_structure_agent(), name="structure_profile")
    bank_node = node(build_questionbank_agent(), name="question_bank")
    interview_node = node(build_interview_agent(), name="interview")
    dossier_node = node(build_dossier_agent(), name="build_dossier")
    return Workflow(
        name="repo_interview_coach",
        description="Interviews a candidate on their own public GitHub project and produces an interview-prep dossier.",
        edges=[(START, load_node, structure_node, bank_node, interview_node, dossier_node)],
    )


root_agent = build_workflow()
