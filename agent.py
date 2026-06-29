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

from google.adk.agents import LlmAgent
from google.adk.tools._request_input_tool import request_input
from google.adk.workflow import START, Workflow, node

from .github_tools import (
    fetch_file_tree,
    fetch_languages,
    fetch_readme,
    fetch_recent_commits,
    fetch_repo_overview,
)
from .guardrails import sanitize_github_args
from .schemas import Dossier, ProjectProfile

MODEL = "gemini-2.5-flash"

_LOAD_INSTRUCTION = """\
The user gives you a public GitHub project as "owner/repo" or a GitHub URL.
Parse the owner and repo. Call fetch_repo_overview first; if it errors (private
or not found), say so plainly and stop. Otherwise call fetch_readme,
fetch_recent_commits, fetch_file_tree and fetch_languages.

Then write a thorough, plain-text briefing of the project grounded ONLY in what
the tools returned: what it does, its languages, the notable technical
components/modules visible in the file tree, what recent commits show was built,
and the weak spots a sharp interviewer would probe. Never invent anything."""

_STRUCTURE_INSTRUCTION = """\
Here is a briefing of a GitHub project:

{raw_context}

Produce a structured ProjectProfile from it. Use only facts present in the
briefing. If the project is thin, reflect that honestly in likely_weak_spots."""

_INTERVIEW_INSTRUCTION = """\
You are a sharp, demanding technical interviewer. The candidate built this project:

{profile}

Conduct a focused interview of 5 to 6 questions. Use the request_input tool to
ask EACH question and wait for the candidate's answer — one question per
request_input call, never batch them. After each answer, briefly note (to
yourself) a 1-5 score, what was strong, and what was missing, then ask the next,
harder question that probes a weak spot from the profile.

Cover, roughly in order: a 60-second project pitch, an architecture/design
decision, the hardest bug or tradeoff, a "why X over Y" choice, and scaling or
failure modes. Be direct, no flattery. If the candidate bluffs or hand-waves,
call it out and dig deeper.

When the interview is done, output a plain-text transcript with one block per
question, each in EXACTLY this format so the next step can parse it:

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


def build_load_agent() -> LlmAgent:
    """Context Loader — calls the (guarded) GitHub tools, emits rich plain text."""
    return LlmAgent(
        name="load_context",
        model=MODEL,
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
        model=MODEL,
        description="Distils the raw project briefing into a typed ProjectProfile.",
        instruction=_STRUCTURE_INSTRUCTION,
        output_schema=ProjectProfile,
        output_key="profile",
    )


def build_interview_agent() -> LlmAgent:
    """Live, adaptive HITL interview via the request_input long-running tool."""
    return LlmAgent(
        name="interview",
        model=MODEL,
        description="Conducts a live, adaptive technical interview about the project.",
        instruction=_INTERVIEW_INSTRUCTION,
        tools=[request_input],
        output_key="transcript",
    )


def build_dossier_agent() -> LlmAgent:
    """Builds the final typed Dossier."""
    return LlmAgent(
        name="build_dossier",
        model=MODEL,
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
