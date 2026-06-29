"""Guardrail callback for the GitHub tools.

Wired as the Context Loader's `before_tool_callback`. It runs before every
GitHub tool call and:
- normalises owner/repo (handles a pasted URL or "owner/repo" in one field),
- enforces GitHub's name charset so we can't be tricked into a weird path,
- clamps numeric args to safe ranges (no 10,000-commit or 500k-char pulls),
- blocks the call with a clean error if owner/repo can't be salvaged.

Returning None lets the (possibly-mutated) call proceed; returning a dict
short-circuits the tool with that result.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_URL_RE = re.compile(r"github\.com[/:]+([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


def _clamp(args: dict[str, Any], key: str, lo: int, hi: int) -> None:
    if key in args:
        try:
            args[key] = max(lo, min(int(args[key]), hi))
        except (TypeError, ValueError):
            args[key] = hi


def sanitize_github_args(tool: Any, args: dict[str, Any], tool_context: Any) -> Optional[dict]:
    """before_tool_callback: clean + clamp args for the GitHub fetch tools.

    ADK invokes this with keyword args (tool=, args=, tool_context=), so the
    third parameter must be named `tool_context`.
    """
    owner = str(args.get("owner", "")).strip()
    repo = str(args.get("repo", "")).strip()

    # A full GitHub URL or "owner/repo" pasted into either field.
    blob = f"{owner} {repo}".strip()
    m = _URL_RE.search(blob)
    if m:
        owner, repo = m.group(1), m.group(2)
    elif "/" in owner and not repo:
        owner, _, repo = owner.partition("/")

    owner = owner.strip().strip("/")
    repo = repo.strip().strip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]

    if not _OWNER_RE.match(owner) or not _REPO_RE.match(repo):
        return {
            "error": (
                f"Refusing to call {getattr(tool, 'name', 'tool')}: could not parse a valid "
                f"GitHub owner/repo from owner={args.get('owner')!r} repo={args.get('repo')!r}. "
                "Ask the user for a public repo as 'owner/repo'."
            )
        }

    args["owner"], args["repo"] = owner, repo
    _clamp(args, "limit", 1, 30)
    _clamp(args, "max_files", 1, 200)
    _clamp(args, "max_chars", 500, 20000)
    return None
