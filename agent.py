"""Repo Interview Coach — root agent.

Phase 1: a tool-calling Context Loader. Given a public GitHub repo (as
"owner/repo" or a URL), it pulls the real README, commits, file tree and
languages and reports a grounded profile of the project.

Phase 2 will wrap this in a Workflow graph (load -> interview LoopAgent with
RequestInput HITL -> Dossier builder). Kept as a plain tool-calling LlmAgent
for now because an agent with an output_schema cannot also call tools — the
typed ProjectProfile is produced by a downstream structuring node in Phase 2.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent

from .github_tools import (
    fetch_file_tree,
    fetch_languages,
    fetch_readme,
    fetch_recent_commits,
    fetch_repo_overview,
)

MODEL = "gemini-2.5-flash"

_LOADER_INSTRUCTION = """\
You are the Context Loader for an interview-prep coach. The user gives you a
public GitHub project as "owner/repo" or a GitHub URL.

Your job:
1. Parse the owner and repo from what the user gave you.
2. Call fetch_repo_overview first to confirm it exists. If it returns an error
   (private, not found), tell the user plainly and stop.
3. Then call fetch_readme, fetch_recent_commits, fetch_file_tree and
   fetch_languages to load the REAL project.
4. Report a grounded profile: what the project does, its main language(s), the
   notable technical components/modules you can see, what recent commits show
   was built, and the likely weak spots a sharp interviewer would probe.

Ground everything in what the tools returned. NEVER invent features, metrics, or
architecture the repo does not show. If the repo is thin, say so plainly.
"""

root_agent = LlmAgent(
    name="root_agent",
    model=MODEL,
    description="Loads a public GitHub project's real content and reports a grounded technical profile for interview prep.",
    instruction=_LOADER_INSTRUCTION,
    tools=[
        fetch_repo_overview,
        fetch_readme,
        fetch_recent_commits,
        fetch_file_tree,
        fetch_languages,
    ],
)
