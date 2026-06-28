# Repo Interview Coach

An AI-powered interview preparation coach built with **Google Agent Development Kit (ADK)** that analyzes any public GitHub repository and gives you a grounded technical profile to ace your interviews.

## What it does

Given a public GitHub repository (`owner/repo` or a GitHub URL), the agent:

1. Fetches the real README, recent commits, file tree, and language stats
2. Reports a grounded technical profile: what the project does, its main languages, notable components, and recent development activity
3. Highlights the **weak spots a sharp interviewer would probe** — so you're never caught off guard

> Ground truth only. It never invents features, metrics, or architecture the repo doesn't show.

## Project Structure

```
.
├── agent.py          # Root LlmAgent (Context Loader)
├── github_tools.py   # GitHub API tool functions
├── schemas.py        # Pydantic schemas (ProjectProfile etc.)
├── __init__.py
└── .env              # GitHub token (not committed)
```

- **Phase 1 (current):** Tool-calling `LlmAgent` that loads and profiles the repo
- **Phase 2 (planned):** Full Workflow graph — `load → interview LoopAgent (HITL) → Dossier builder`

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash |
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

- [x] Phase 1: Context Loader (tool-calling agent)
- [ ] Phase 2: Interactive interview loop (HITL with `RequestInput`)
- [ ] Phase 3: Dossier builder (structured output)
- [ ] Phase 4: Multi-repo comparison mode

## License

MIT
