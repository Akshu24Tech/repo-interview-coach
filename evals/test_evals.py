"""Pytest gate wrapping the eval suite.

    uv run pytest evals/test_evals.py

Both tests hit the live model + GitHub, so they need GOOGLE_API_KEY (loaded from
repo_interview_coach/.env) and network access.
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

# Make the suite importable/runnable from inside the package dir (see
# eval_harness for the rationale): package parent on path, .env by abs path.
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PKG_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
load_dotenv(os.path.join(_PKG_DIR, ".env"))


def test_deterministic_harness_all_pass():
    from evals.eval_harness import main

    assert asyncio.run(main()) == 0, "deterministic eval harness had failures"


def test_official_adk_evalset_passes():
    from google.adk.evaluation import AgentEvaluator

    asyncio.run(
        AgentEvaluator.evaluate(
            agent_module="repo_interview_coach.analysis_app",
            eval_dataset_file_path_or_dir="evals/loader.evalset.json",
            num_runs=1,
            print_detailed_results=False,
        )
    )  # raises AssertionError if thresholds are not met
