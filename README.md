# Repo Interview Coach

An AI-powered interview preparation coach built with **Google Agent Development Kit (ADK)** that analyzes any public GitHub repository and gives you a grounded technical profile to ace your interviews.

**Live demo:** https://repo-interview-coach-788528493953.us-central1.run.app/dev-ui/ (Cloud Run)

## What it does

Given a public GitHub repository (`owner/repo` or a GitHub URL), the agent:

1. **Fetches** the real README, recent commits, file tree, and language stats
2. **Profiles** it: what the project does, its main languages, notable components, recent work, and the **weak spots a sharp interviewer would probe**
3. **Builds a project-specific question bank** — tough questions that only make sense for *this* repo
4. **Interviews you** live, in rounds of 3: it asks one question at a time, you answer, then it reviews the round (score + strength/gap per answer) and asks whether you want another round. Say no and it stops; say yes for more (capped so it can never loop)
5. **Hands you a dossier**: STAR-shaped resume bullets, a refined question bank, and an upgrade list

> Ground truth only. It never invents features, metrics, or architecture the repo doesn't show.

## The interview loop

The interview is driven by a **deterministic state machine in code**, not by trusting the model to
remember what it asked — a self-looping LLM re-enters fresh on each human-in-the-loop pause and weaker
models just re-ask the opener. Answers are counted from the authoritative session history, and the
question tool is physically removed once the interview ends, so **it stops cleanly on any model**. The
batching/review/continue/stop logic has its own deterministic eval (see `evals/`).

## Project Structure

```
.
├── agent.py               # The full ADK Workflow: nodes, guardrail wiring, interview state machine
├── github_tools.py        # GitHub API tool functions (graceful failure)
├── guardrails.py          # before_tool callback: clean/clamp/block tool args
├── schemas.py             # Pydantic schemas (ProjectProfile, QuestionBank, Dossier, AnswerScore)
├── analysis_app/          # The deterministic load→structure half, as an eval target
├── evals/                 # 29-case, two-tier eval suite (see evals/README.md)
├── __init__.py
└── .env                   # model + GitHub token (not committed)
```

ADK Workflow graph: `load_context → structure_profile → question_bank → interview (HITL) → build_dossier`,
with a `before_tool` guardrail and a green eval suite.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Configurable via `RIC_MODEL` (local Ollama for free unlimited runs; Gemini / OpenRouter / any LiteLLM provider for the cloud) |
| Data Source | GitHub REST API |
| Schema Validation | Pydantic |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Akshu24Tech/repo-interview-coach.git
cd repo-interview-coach

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install google-adk

# 4. Add your GitHub token (optional, for higher rate limits)
echo "GITHUB_TOKEN=your_token_here" > .env

# 5. Run the agent
adk run .
```

## Usage

Once running, provide a GitHub repo:

```
> google/generative-ai-python
```

The agent fetches real content and gives you a technical profile with interview talking points.

## Roadmap

- [x] Phase 1: Context Loader (tool-calling agent) + schemas + GitHub tools
- [x] Phase 2: Workflow graph + interactive interview loop (HITL `request_input`)
- [x] Phase 3: `before_tool` guardrail + answer scoring
- [x] Phase 4: Eval suite — **29 cases, two-tier** (deterministic gate + live GitHub/model tier) + ADK AgentEvaluator
- [x] Phase 5: Deploy to Cloud Run (`--with_ui`, max-instances=1) + verified live link
- [x] Phase 6: **Batched interview** — rounds of 3 with a review + "want more?" checkpoint, driven by a code state machine; project-specific **question bank**
- [ ] Phase 7: README/talk-track + resume bullet; redeploy the batched build to Cloud Run

## License

MIT
