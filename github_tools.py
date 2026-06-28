"""GitHub fetch tools for the Repo Interview Coach.

Plain functions -> ADK wraps each as a FunctionTool when passed in an agent's
`tools` list. They hit the PUBLIC GitHub REST API. No token is required
(unauthenticated = 60 req/hr); if GITHUB_TOKEN is set in the environment it is
used to raise the limit. Every function returns a string or JSON-serializable
value and degrades gracefully (returns an error string instead of raising) so
the agent can react instead of crashing.
"""

from __future__ import annotations

import os

import requests

_API = "https://api.github.com"
_TIMEOUT = 15
_UA = "repo-interview-coach"


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"User-Agent": _UA, "Accept": accept}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get(url: str, accept: str = "application/vnd.github+json"):
    try:
        resp = requests.get(url, headers=_headers(accept), timeout=_TIMEOUT)
    except requests.RequestException as exc:
        return None, f"ERROR: request failed: {exc}"
    if resp.status_code == 404:
        return None, "ERROR: not found (repo may be private, renamed, or misspelled)."
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        return None, "ERROR: GitHub rate limit hit. Try again later or set GITHUB_TOKEN."
    if resp.status_code >= 400:
        return None, f"ERROR: GitHub returned {resp.status_code}."
    return resp, None


def fetch_repo_overview(owner: str, repo: str) -> dict:
    """Fetch high-level metadata for a public GitHub repo: description, primary
    language, stars, default branch, and topics. Call this first to confirm the
    repo exists and get its shape.

    Args:
        owner: GitHub username or org, e.g. "Akshu24Tech".
        repo: repository name, e.g. "repo-guardian".
    """
    resp, err = _get(f"{_API}/repos/{owner}/{repo}")
    if err:
        return {"error": err}
    d = resp.json()
    return {
        "name": d.get("name"),
        "description": d.get("description") or "",
        "primary_language": d.get("language") or "unknown",
        "stars": d.get("stargazers_count", 0),
        "default_branch": d.get("default_branch", "main"),
        "topics": d.get("topics", []),
        "pushed_at": d.get("pushed_at"),
    }


def fetch_readme(owner: str, repo: str, max_chars: int = 6000) -> str:
    """Fetch the raw README text of a public GitHub repo (truncated). Use this
    to understand what the project does and how it is structured.

    Args:
        owner: GitHub username or org.
        repo: repository name.
        max_chars: maximum characters to return.
    """
    resp, err = _get(f"{_API}/repos/{owner}/{repo}/readme", accept="application/vnd.github.raw+json")
    if err:
        return err
    text = resp.text
    return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")


def fetch_recent_commits(owner: str, repo: str, limit: int = 20) -> list[str]:
    """Fetch recent commit messages (with dates) for a public GitHub repo. Use
    this to see what was actually built and changed.

    Args:
        owner: GitHub username or org.
        repo: repository name.
        limit: number of recent commits (max 30).
    """
    limit = max(1, min(limit, 30))
    resp, err = _get(f"{_API}/repos/{owner}/{repo}/commits?per_page={limit}")
    if err:
        return [err]
    out = []
    for c in resp.json():
        commit = c.get("commit", {})
        msg = (commit.get("message") or "").splitlines()[0]
        date = (commit.get("author") or {}).get("date", "")[:10]
        out.append(f"{date}: {msg}")
    return out


def fetch_file_tree(owner: str, repo: str, max_files: int = 60) -> list[str]:
    """Fetch the file paths in a public GitHub repo (recursive, truncated). Use
    this to understand the project's structure and spot key modules.

    Args:
        owner: GitHub username or org.
        repo: repository name.
        max_files: maximum number of paths to return.
    """
    ov = fetch_repo_overview(owner, repo)
    if "error" in ov:
        return [ov["error"]]
    branch = ov.get("default_branch", "main")
    resp, err = _get(f"{_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
    if err:
        return [err]
    paths = [t["path"] for t in resp.json().get("tree", []) if t.get("type") == "blob"]
    return paths[:max_files]


def fetch_languages(owner: str, repo: str) -> dict:
    """Fetch the language breakdown (language -> bytes) for a public GitHub repo.

    Args:
        owner: GitHub username or org.
        repo: repository name.
    """
    resp, err = _get(f"{_API}/repos/{owner}/{repo}/languages")
    if err:
        return {"error": err}
    return resp.json()


if __name__ == "__main__":
    # Smoke test against a real public repo.
    O, R = "Akshu24Tech", "repo-guardian"
    print("== overview =="); print(fetch_repo_overview(O, R))
    print("== languages =="); print(fetch_languages(O, R))
    print("== commits (first 5) =="); print("\n".join(fetch_recent_commits(O, R)[:5]))
    print("== file tree (first 15) =="); print("\n".join(fetch_file_tree(O, R)[:15]))
    rm = fetch_readme(O, R)
    print("== readme (first 300 chars) =="); print(rm[:300])
