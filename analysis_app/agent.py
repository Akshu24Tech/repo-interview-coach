"""Eval target (ADK package convention): the deterministic analysis half.

The full Workflow includes a human-in-the-loop interview (request_input), which
does not fit scripted evaluation. So `adk eval` / AgentEvaluator runs against the
gradeable part: load_context -> structure_profile (fetch real repo data ->
typed ProjectProfile).

    adk eval repo_interview_coach/analysis_app evals/loader.evalset.json
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from ..agent import build_load_agent, build_structure_agent

root_agent = SequentialAgent(
    name="repo_analysis",
    description="Loads a public GitHub repo and produces a typed ProjectProfile.",
    sub_agents=[build_load_agent(), build_structure_agent()],
)
