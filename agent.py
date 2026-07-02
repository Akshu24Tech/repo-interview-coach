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
from .schemas import Dossier, ProjectProfile

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

Conduct a focused technical interview, ONE question per request_input call,
never batched. A [INTERVIEWER CONTROL] message will tell you exactly how many
questions there are in total, which numbered question to ask next, and the topic
it must cover — follow it exactly and never repeat an earlier question. Tailor
each question to THIS candidate's project and make it harder than the last; probe
a real weak spot from the profile. Be direct, no flattery. If the candidate
bluffs or hand-waves, call it out.

When the control message says all questions have been answered, STOP calling
request_input and output a plain-text transcript with one block PER QUESTION YOU
ACTUALLY ASKED (do not invent extra questions or answers), each in EXACTLY this
format so the next step can parse it:

Q: <the question>
A: <the candidate's answer, summarised>
Score: <N>/5
Strength: <one line>
Gap: <one line — what was missing or wrong>
Model answer: <one tight line grounded in the project>

This transcript is for the next step."""

_DOSSIER_INSTRUCTION = """\
Project profile:

{profile}

Interview transcript:

{transcript}

Produce the final Dossier: 3-4 STAR-shaped resume bullets (honest, quantified
only where the project supports it), a question bank of likely interview
questions, an upgrade list of concrete improvements that would both strengthen
the project and give better interview stories, and a short honest read on how
interview-ready it is."""


# Fixed interview plan. Progression is driven by code (counting answered
# questions), not by trusting the model to remember what it already asked — a
# single self-looping LlmAgent re-enters fresh on each request_input resume and
# weaker models just re-ask the opener. See DEBUG-LOG 2026-06-30.
_INTERVIEW_THEMES = [
    "a 60-second pitch of what the project does and why it matters",
    "a key architecture or design decision, and the reasoning behind it",
    "the hardest bug, tradeoff, or failure they hit and how they handled it",
]
_REQUEST_INPUT_NAME = "adk_request_input"


def _count_answered(callback_context, llm_request) -> int:
    """Number of interview answers so far = count of `adk_request_input`
    function_responses. Counted from the full session event history
    (callback_context.session.events), which is the authoritative record — the
    per-turn `llm_request.contents` is rebuilt on each Workflow resume and does
    NOT reliably carry prior answers, so counting it stuck at 0 and the interview
    re-asked question #1 forever. Falls back to contents if session is absent."""
    def _tally(iterable) -> int:
        n = 0
        for item in iterable or []:
            content = getattr(item, "content", item)
            for part in getattr(content, "parts", None) or []:
                fr = getattr(part, "function_response", None)
                if fr is not None and fr.name == _REQUEST_INPUT_NAME:
                    n += 1
        return n

    session = getattr(callback_context, "session", None)
    events = getattr(session, "events", None)
    if events:
        return _tally(events)
    return _tally(llm_request.contents)


def _steer_interview(callback_context, llm_request):
    """before_model_callback: count answered interview questions from history and
    inject a hard [INTERVIEWER CONTROL] directive naming the exact next question
    to ask (or to stop). Makes progression deterministic on any model."""
    answered = _count_answered(callback_context, llm_request)

    if answered >= len(_INTERVIEW_THEMES):
        directive = (
            f"All {len(_INTERVIEW_THEMES)} questions have been asked and answered. "
            "Do NOT call request_input again. Output the final plain-text transcript "
            "now, one block per question, in the exact required format."
        )
        # Deterministic stop: physically remove the request_input tool so a weak
        # model CANNOT keep asking. With no tool to call, it must emit the final
        # transcript text. Instruction alone doesn't hold on small models (see
        # DEBUG-LOG: llama-3.3/scout looped even when told to stop).
        cfg = getattr(llm_request, "config", None)
        if cfg is not None and cfg.tools:
            for tool in cfg.tools:
                fds = getattr(tool, "function_declarations", None)
                if fds:
                    tool.function_declarations = [
                        fd for fd in fds if fd.name != _REQUEST_INPUT_NAME
                    ]
            cfg.tools = [
                t for t in cfg.tools if getattr(t, "function_declarations", None)
            ]
        if getattr(llm_request, "tools_dict", None):
            llm_request.tools_dict.pop(_REQUEST_INPUT_NAME, None)
    else:
        done = ", ".join(
            f"#{i + 1} ({_INTERVIEW_THEMES[i].split(',')[0]})" for i in range(answered)
        ) or "none yet"
        directive = (
            f"Questions already asked and answered: {done}. "
            f"Now ask question #{answered + 1} of {len(_INTERVIEW_THEMES)}, which MUST cover: "
            f"{_INTERVIEW_THEMES[answered]}. Ask exactly ONE new question via request_input, "
            "specific to this candidate's project and harder than the last. Never repeat a prior question."
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
    """Wire the four agents into the interview-coach graph."""
    load_node = node(build_load_agent(), name="load_context")
    structure_node = node(build_structure_agent(), name="structure_profile")
    interview_node = node(build_interview_agent(), name="interview")
    dossier_node = node(build_dossier_agent(), name="build_dossier")
    return Workflow(
        name="repo_interview_coach",
        description="Interviews a candidate on their own public GitHub project and produces an interview-prep dossier.",
        edges=[(START, load_node, structure_node, interview_node, dossier_node)],
    )


root_agent = build_workflow()
