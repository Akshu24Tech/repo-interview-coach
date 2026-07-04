"""Typed contracts for the Repo Interview Coach.

These Pydantic models are the structured I/O for the agent graph:
- ProjectProfile  -> what the Context Loader distils from a repo
- AnswerScore     -> what the Evaluator returns for each interview answer
- Dossier         -> the final deliverable handed back to the candidate
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectProfile(BaseModel):
    """A grounded summary of a GitHub project, built from its real README,
    commits, file tree and languages. Everything here must come from the repo,
    never invented."""

    owner: str = Field(description="GitHub username/org that owns the repo.")
    repo: str = Field(description="Repository name.")
    summary: str = Field(description="What the project actually does, in 2-3 sentences, grounded in the README.")
    primary_language: str = Field(description="Main programming language.")
    languages: list[str] = Field(default_factory=list, description="All notable languages used.")
    notable_components: list[str] = Field(default_factory=list, description="Key modules, services, agents, or files that matter technically.")
    recent_work: list[str] = Field(default_factory=list, description="What recent commits show was built or changed.")
    likely_weak_spots: list[str] = Field(default_factory=list, description="Areas a sharp interviewer would probe: missing tests, no error handling, unclear scaling, thin docs, etc.")


class QuestionBank(BaseModel):
    """A project-specific set of tough interview questions, generated from the
    profile BEFORE the interview and reused by the dossier. Ordered warm-up →
    hardest. This is what makes the follow-up rounds sharp and grounded instead
    of generic."""

    owner: str = Field(description="GitHub username/org that owns the repo.")
    repo: str = Field(description="Repository name.")
    questions: list[str] = Field(
        description="8-10 sharp, project-specific interview questions, ordered from a "
        "warm-up to the hardest. Each must only make sense for THIS project — grounded "
        "in its real components, recent work, and likely weak spots."
    )


class AnswerScore(BaseModel):
    """The Evaluator's verdict on one interview answer."""

    question: str = Field(description="The question that was asked.")
    score: int = Field(ge=1, le=5, description="Quality of the candidate's answer, 1 (poor) to 5 (excellent).")
    strengths: str = Field(description="What was strong about the answer.")
    gaps: str = Field(description="What was missing, vague, or wrong.")
    model_answer: str = Field(description="A tight, ideal answer grounded in the project's real code.")


class Dossier(BaseModel):
    """The final interview-prep deliverable for one project."""

    owner: str
    repo: str
    resume_bullets: list[str] = Field(description="3-4 STAR-shaped resume bullets, honest, quantified only where the repo supports it.")
    question_bank: list[str] = Field(description="Likely interview questions for this project, behavioral + technical + system-design.")
    upgrade_list: list[str] = Field(description="Concrete additions that would both strengthen the project and give better interview stories (tests, eval, demo, observability, security).")
    overall_notes: str = Field(description="Short honest read on how interview-ready this project is.")
