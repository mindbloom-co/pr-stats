#!/usr/bin/env python3
"""
Mindbloom PR Metrics Calculator

Calculates per-person PR authoring & reviewing stats across repos,
broken into 2-week cycles with cycle-over-cycle comparison.

Metrics:
  - PRs authored (merged) per person per cycle
  - PRs opened per person per cycle
  - Reviews given per person per cycle
  - Time to merge (created → merged) per person and team average (human-authored PRs only)

PRs/engineer KPI: human-authored merged PRs plus merged Devin PRs credited to
their first human approver (bots and EXCLUDED_USERS never count as approvers).
Devin PRs with no human approval are excluded and reported as a count.

Usage:
    python pr-metrics.py                        # last 1 cycle (2 weeks)
    python pr-metrics.py 4                      # last 4 cycles (8 weeks)
    python pr-metrics.py 2025-01-01 2025-06-01  # custom date range

    # Live from GitHub, with frozen Q1-Q2 history (the normal refresh)
    python pr-metrics.py 2026-01-01 2026-11-04 \\
        --history pr-metrics-history.json --collect-from 2026-07-01 \\
        --as-of "2026-11-04 ~9:30 AM CT"

    # Optional fallback without gh: add --input FILE (repeatable) with pre-collected PRs

--input (optional, repeatable) takes a JSON list of merged PRs ({author,
created/createdAt/created_at, merged/mergedAt/merged_at}, optional repo,
n/number, reviewers, is_ai_pr, shepherd/credited_to, all_approvals, labels), or an
object {"merged": [...], "opened": [...]}. Devin PRs are credited from
all_approvals ([login, state, submitted_at]) when present. Metrics the input
does not contain (opened, reviews, AI PRs) are reported as not collected.

--history takes frozen per-cycle aggregates for earlier cycles; collection
(live or --input) starts the day after the last frozen cycle, or at
--collect-from. Frozen cycles do not get Devin crediting (no per-PR reviews).

Autonomous merges (separate from the KPIs above): share of merged PRs labeled
"autonomous-merge" among all merged PRs, bots, Dependabot and Devin included
(EXCLUDED_USERS left out), since 2026-10-21. Input records may carry "labels"
(missing = none) and all_approvals for the human-approval cross-check. See README.md.

Requires: gh (GitHub CLI) authenticated with org access, unless --input is used
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# ── Configuration ───────────────────────────────────────────

ORG = "mindbloom-co"
REPOS = [
    "mindbloomMobile",
    "mindbloomBackend",
    "mindbloomFrontend",
    "mindbloomBackOffice",
    "infrastructure",
    "mindbloomDbt",
    "mindbloom-data",
    "mindbloom-data-mcp",
]
# Effective-start date per repo (YYYY-MM-DD). A repo listed here is not queried and not counted
# for any PR merged (or opened) before its date. Repos not listed count from the start.
REPO_EFFECTIVE_START = {
    "mindbloomDbt": "2026-10-01",       # Q4'26
    "mindbloom-data": "2026-10-01",     # Q4'26
    "mindbloom-data-mcp": "2026-10-01", # Q4'26
}
REPOS_NOTE = "mindbloomDbt, mindbloom-data and mindbloom-data-mcp included from Q4'26 (2026-10-01); earlier quarters unchanged."
CYCLE_DAYS = 14

# Goals as set at the time; the PR KPI started 2026-04-17, so Q1 has none.
QUARTERLY_PRS_PER_ENGINEER_TARGETS = {
    "Q2 2026": 90,
    "Q3 2026": 110,
    "Q4 2026": 140,
}
TTM_P90_TARGET_HOURS = {
    "Q4 2026": 72,
}
TTM_PERCENTILES = (50, 75, 90, 95)

# Bot accounts to exclude from all stats
BOTS = {
    "gemini-code-assist",
    "devin-ai-integration",
    "github-actions",
    "github-advanced-security",
    "dependabot",
    "claude",
    "coderabbitai",
    "chatgpt-codex-connector",
    "blacksmith-sh",
}

# Human accounts to exclude from current team KPI reporting
EXCLUDED_USERS = {
    "gracesliu",
}

# AI coding agents — their PRs get attributed to the human who approved them
AI_AGENTS = {"devin-ai-integration"}

# Autonomous merges: PRs carrying this label were authored, reviewed and queued by agents.
# Unlike the KPIs above, this metric counts bot and Devin PRs (numerator and denominator).
AUTONOMOUS_LABEL = "autonomous-merge"
AUTONOMOUS_TRACKING_START = "2026-10-21"  # first labeled merge: mindbloomBackend #6188
# Reviewers that never count as a human approval in the autonomous cross-check, in
# addition to BOTS and any "[bot]" login. The merge queue app posts as a bot.
NON_HUMAN_REVIEWERS = {"github-merge-queue", "github-merge-queue-bot", "merge-queue"}


# ── GitHub API ──────────────────────────────────────────────

def gh_graphql(query: str) -> dict:
    """Run a GraphQL query via the gh CLI and return parsed JSON."""
    for attempt in range(3):
        result = subprocess.run(
            ["gh", "api", "graphql", "-f", f"query={query}"],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        if result.returncode == 0 and data.get("data") and not data.get("errors"):
            return data
        error = result.stderr.strip() or str(data.get("errors", "Empty GitHub response"))
        if attempt < 2:
            print(f"GitHub request failed; retrying: {error}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GitHub request failed after 3 attempts: {error}")


def _paginated_search(search_query: str, extra_fields: str = "") -> list[dict]:
    for attempt in range(3):
        try:
            return _paginated_search_once(search_query, extra_fields)
        except RuntimeError:
            if attempt == 2:
                raise
            print(f"Incomplete search; retrying: {search_query}", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise RuntimeError("Search retries exhausted")


def _paginated_search_once(search_query: str, extra_fields: str = "") -> list[dict]:
    """Generic paginated GitHub search returning PR nodes."""
    count_data = gh_graphql(f'{{ search(query: "{search_query}", type: ISSUE, first: 1) {{ issueCount }} }}')
    expected_count = (count_data["data"].get("search") or {}).get("issueCount")
    if expected_count is None:
        raise RuntimeError(f"Missing search count: {search_query}")
    results = []
    has_next = True
    cursor = None

    while has_next:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""{{
            search(query: "{search_query}", type: ISSUE, first: 100{after}) {{
                issueCount
                pageInfo {{ hasNextPage endCursor }}
                nodes {{
                    ... on PullRequest {{
                        author {{ login }}
                        createdAt
                        mergedAt
                        {extra_fields}
                    }}
                }}
            }}
        }}"""

        data = gh_graphql(query)
        search_data = data["data"].get("search")
        if not search_data or not {"issueCount", "pageInfo", "nodes"} <= search_data.keys():
            raise RuntimeError(f"Invalid search response: {search_query}")
        page_info = search_data["pageInfo"]
        results.extend(search_data["nodes"])

        has_next = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")
        if has_next and not cursor:
            raise RuntimeError(f"Missing search cursor: {search_query}")

    if len(results) != search_data["issueCount"] or len(results) != expected_count:
        raise RuntimeError(f"Incomplete search results: {search_query}")

    return results


def fetch_merged_prs(repo: str, start: str, end: str) -> list[dict]:
    """
    Fetch all merged PRs for a repo in a date range.
    Returns list of {repo, n, author, reviewers, created_at, merged_at, ttm_hours, labels, all_approvals}
    """
    search_query = f"repo:{ORG}/{repo} is:pr is:merged merged:{start}..{end}"
    review_fields = """
        number
        labels(first: 20) { totalCount nodes { name } }
        reviews(first: 50) {
            totalCount
            nodes {
                author { login }
                state
                submittedAt
            }
        }
    """
    nodes = _paginated_search(search_query, review_fields)
    prs = []

    for node in nodes:
        author = (node.get("author") or {}).get("login", "unknown")

        # Reviewers
        review_states = {"APPROVED", "CHANGES_REQUESTED", "COMMENTED"}
        reviewers = set()
        all_approvals = []
        reviews = node.get("reviews") or {}
        if reviews.get("totalCount", 0) > len(reviews.get("nodes", [])):
            print(f"  ⚠️  {repo}#{node.get('number')}: more than 50 reviews; extra reviews ignored", file=sys.stderr)
        labels = node.get("labels") or {}
        if labels.get("totalCount", 0) > len(labels.get("nodes", [])):
            print(f"  ⚠️  {repo}#{node.get('number')}: more than 20 labels; extra labels ignored", file=sys.stderr)
        for review in reviews.get("nodes", []):
            reviewer = (review.get("author") or {}).get("login")
            if reviewer and review.get("state") in review_states:
                reviewers.add(reviewer)
            if reviewer and review.get("state") == "APPROVED":
                all_approvals.append([reviewer, "APPROVED", review.get("submittedAt")])

        # For AI agent PRs, attribute to the first human approver
        is_ai_pr = is_ai_agent(author)
        shepherd = first_human_approver(all_approvals) if is_ai_pr else None

        created_at = node.get("createdAt")
        merged_at = node.get("mergedAt")
        prs.append({
            "repo": repo,
            "n": node.get("number"),
            "author": author,
            "reviewers": list(reviewers),
            "shepherd": shepherd,
            "is_ai_pr": is_ai_pr,
            "created_at": created_at,
            "merged_at": merged_at,
            "ttm_hours": hours_between(created_at, merged_at),
            "labels": [label["name"] for label in labels.get("nodes", [])],
            "all_approvals": all_approvals,
        })

    return prs


def fetch_opened_prs(repo: str, start: str, end: str) -> list[dict]:
    """
    Fetch all PRs created in a date range (regardless of merge status).
    Returns list of {author}
    """
    search_query = f"repo:{ORG}/{repo} is:pr created:{start}..{end}"
    nodes = _paginated_search(search_query)
    return [
        {"author": (node.get("author") or {}).get("login", "unknown")}
        for node in nodes
    ]


# ── Helpers ─────────────────────────────────────────────────

def base_login(username: str) -> str:
    """GraphQL returns bot logins bare; REST exports append "[bot]"."""
    return username.removesuffix("[bot]")


def is_human(username: str) -> bool:
    base = base_login(username)
    return not username.endswith("[bot]") and base not in BOTS and base not in EXCLUDED_USERS


def is_ai_agent(username: str) -> bool:
    return base_login(username) in AI_AGENTS


def first_human_approver(approvals: list) -> str | None:
    """approvals: [login, state, submitted_at] triples, as in the Devin credit exports."""
    approved = sorted(
        (a for a in approvals if a[1] == "APPROVED" and is_human(a[0])),
        key=lambda a: a[2],
    )
    return approved[0][0] if approved else None


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def repo_window(repo: str, start: str, end: str) -> tuple[str, str] | None:
    """Clamp a YYYY-MM-DD window to the repo's effective start; None when it ends before it."""
    first = REPO_EFFECTIVE_START.get(repo)
    if first is None or start >= first:
        return start, end
    return (first, end) if end >= first else None


def repo_active(repo: str | None, on_date: datetime) -> bool:
    first = REPO_EFFECTIVE_START.get(repo)
    return first is None or fmt_date(on_date) >= first


def hours_between(created_at: str | None, merged_at: str | None) -> float | None:
    if not (created_at and merged_at):
        return None
    return (parse_ts(merged_at) - parse_ts(created_at)).total_seconds() / 3600


def percentile(values: list[float], pct: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def load_input_prs(paths: list[str]) -> dict:
    """Load pre-collected PRs from one or more files.

    Returns merged/opened lists plus which metrics they cover.
    """
    merged_raw, opened_raw = [], None
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        merged_raw.extend(data["merged"] if isinstance(data, dict) else data)
        if isinstance(data, dict) and data.get("opened") is not None:
            opened_raw = (opened_raw or []) + data["opened"]

    def field(record: dict, *names: str):
        return next((record[n] for n in names if record.get(n) is not None), None)

    def shepherd_for(record: dict) -> str | None:
        if "all_approvals" not in record:
            return field(record, "shepherd", "credited_to")
        shepherd = first_human_approver(record["all_approvals"])
        if "credited_to" in record and record["credited_to"] != shepherd:
            print(
                f"  ⚠️  {record.get('repo')}#{record.get('n')}: credited_to={record['credited_to']} "
                f"but first human approval is {shepherd}; using {shepherd}",
                file=sys.stderr,
            )
        return shepherd

    def label_names(record: dict) -> list[str]:
        """Labels as names; entries may be strings or {"name": ...}. Missing = no labels."""
        return [l["name"] if isinstance(l, dict) else l for l in record.get("labels") or []]

    merged, seen = [], {}
    for record in merged_raw:
        number = field(record, "n", "number")
        if number is not None:
            key = (record.get("repo"), number)
            if key in seen:
                # First record wins for every KPI field. Only the autonomous-merge
                # fields are filled in from a later duplicate (older exports lack them).
                first = seen[key]
                first["labels"] = sorted(set(first["labels"]) | set(label_names(record)))
                if first["all_approvals"] is None and "all_approvals" in record:
                    first["all_approvals"] = record["all_approvals"]
                continue
        created_at = field(record, "created_at", "createdAt", "created")
        merged_at = field(record, "merged_at", "mergedAt", "merged")
        is_ai_pr = record.get("is_ai_pr", is_ai_agent(record["author"]))
        pr = {
            "repo": record.get("repo"),
            "n": number,
            "author": record["author"],
            "reviewers": record.get("reviewers", []),
            "shepherd": shepherd_for(record) if is_ai_pr else None,
            "is_ai_pr": is_ai_pr,
            "created_at": created_at,
            "merged_at": merged_at,
            "ttm_hours": hours_between(created_at, merged_at),
            "labels": label_names(record),
            "all_approvals": record.get("all_approvals"),  # None = approvals not collected
        }
        merged.append(pr)
        if number is not None:
            seen[key] = pr
    opened = None
    if opened_raw is not None:
        opened = [
            {"author": r["author"], "created_at": field(r, "created_at", "createdAt", "created")}
            for r in opened_raw
        ]
    return {
        "merged": merged,
        "opened": opened,
        "reviews_collected": bool(merged_raw) and all("reviewers" in r for r in merged_raw),
        "ai_collected": any(pr["is_ai_pr"] for pr in merged) or any("is_ai_pr" in r for r in merged_raw),
    }


def in_cycle(timestamp: str | None, cycle_start: datetime, cycle_end: datetime) -> bool:
    return bool(timestamp) and cycle_start.date() <= parse_ts(timestamp).date() <= cycle_end.date()


def summarize_cycle(
    cycle_start: datetime,
    cycle_end: datetime,
    merged_prs: list[dict],
    opened_prs: list[dict] | None,
    reviews_collected: bool,
    ai_collected: bool,
) -> dict:
    """Aggregate one cycle into the same shape as a frozen history cycle.

    None means the metric was not collected for this cycle.
    """
    human_merged = [pr for pr in merged_prs if is_human(pr["author"])]
    ai_merged = [pr for pr in merged_prs if pr.get("is_ai_pr")]
    human_opened = [pr for pr in opened_prs if is_human(pr["author"])] if opened_prs is not None else None

    authors: dict[str, dict] = defaultdict(lambda: {
        "merged": 0,
        "opened": 0 if human_opened is not None else None,
        "shepherded": 0 if ai_collected else None,
        "reviews": 0 if reviews_collected else None,
        "ttm_avg_hours": None,
    })
    ttm_by_author = defaultdict(list)
    for pr in human_merged:
        authors[pr["author"]]["merged"] += 1
        if pr.get("ttm_hours") is not None:
            ttm_by_author[pr["author"]].append(pr["ttm_hours"])
    for pr in human_opened or []:
        authors[pr["author"]]["opened"] += 1
    if ai_collected:
        for pr in ai_merged:
            if pr.get("shepherd"):
                authors[pr["shepherd"]]["shepherded"] += 1
    total_reviews = None
    if reviews_collected:
        total_reviews = 0
        for pr in merged_prs:
            for reviewer in pr["reviewers"]:
                if is_human(reviewer):
                    authors[reviewer]["reviews"] += 1
                    total_reviews += 1
    for author, ttms in ttm_by_author.items():
        authors[author]["ttm_avg_hours"] = sum(ttms) / len(ttms)

    ttm_values = [pr["ttm_hours"] for pr in human_merged if pr.get("ttm_hours") is not None]
    return {
        "start": fmt_date(cycle_start),
        "end": fmt_date(cycle_end),
        "opened": len(human_opened) if human_opened is not None else None,
        "merged": len(human_merged),
        "ai": len(ai_merged) if ai_collected else None,
        "ai_unapproved": sum(1 for pr in ai_merged if not pr.get("shepherd")) if ai_collected else None,
        "devin_credit": ai_collected,
        "reviews": total_reviews,
        "ttm_avg_hours": avg_ttm(human_merged),
        "ttm_median_hours": median_ttm(human_merged),
        "ttm_values": ttm_values,
        "authors": dict(authors),
    }


def count_sorted(items: list[str]) -> list[tuple[str, int]]:
    """Count occurrences and return sorted (item, count) descending."""
    counts = defaultdict(int)
    for item in items:
        counts[item] += 1
    return sorted(counts.items(), key=lambda x: -x[1])


def build_cycles(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a date range into 2-week cycles."""
    cycles = []
    current = start
    while current <= end:
        _, current_quarter_end = quarter_bounds(current)
        cycle_end = min(current + timedelta(days=CYCLE_DAYS - 1), current_quarter_end, end)
        cycles.append((current, cycle_end))
        current = cycle_end + timedelta(days=1)
    return cycles


def fmt_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def fmt_duration(hours: float) -> str:
    """Format hours into a human-readable string."""
    if hours < 1:
        return f"{hours * 60:.0f}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"


def quarter_sort_key(label: str) -> tuple[int, int]:
    quarter, year = label.split()
    return int(year), int(quarter[1:])


def quarter_bounds(dt: datetime) -> tuple[datetime, datetime]:
    start_month = ((dt.month - 1) // 3) * 3 + 1
    quarter_start = dt.replace(month=start_month, day=1)
    if start_month == 10:
        quarter_end = dt.replace(year=dt.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        quarter_end = dt.replace(month=start_month + 3, day=1) - timedelta(days=1)
    return quarter_start, quarter_end


def avg_ttm(prs: list[dict]) -> float | None:
    """Average time-to-merge in hours for a list of PRs."""
    ttms = [pr["ttm_hours"] for pr in prs if pr.get("ttm_hours") is not None]
    return sum(ttms) / len(ttms) if ttms else None


def median_ttm(prs: list[dict]) -> float | None:
    """Median time-to-merge in hours."""
    ttms = sorted(pr["ttm_hours"] for pr in prs if pr.get("ttm_hours") is not None)
    if not ttms:
        return None
    mid = len(ttms) // 2
    if len(ttms) % 2 == 0:
        return (ttms[mid - 1] + ttms[mid]) / 2
    return ttms[mid]


# ── Autonomous merges ───────────────────────────────────────
# Separate from the PRs/engineer and time-to-merge KPIs: bots, Dependabot and Devin
# PRs count here; only EXCLUDED_USERS are left out.

AUTONOMOUS_AUTHOR_GROUPS = ("linuxlewis", "dependabot", "other")


def autonomous_author_group(author: str) -> str:
    base = base_login(author)
    return base if base in AUTONOMOUS_AUTHOR_GROUPS[:-1] else "other"


def counts_for_autonomous(pr: dict) -> bool:
    return base_login(pr["author"]) not in EXCLUDED_USERS


def is_autonomous(pr: dict) -> bool:
    return AUTONOMOUS_LABEL in (pr.get("labels") or [])


def is_human_reviewer(login: str) -> bool:
    """Human for the approval cross-check: not a bot, Devin or the merge queue app.

    Unlike is_human, EXCLUDED_USERS are people, so their approvals count.
    """
    base = base_login(login)
    return not login.endswith("[bot]") and base not in BOTS and base not in NON_HUMAN_REVIEWERS


def has_human_approval(pr: dict) -> bool | None:
    """True when a human APPROVED the PR no later than its merge; None when approvals were not collected."""
    approvals = pr.get("all_approvals")
    if approvals is None:
        return None
    merged_at = parse_ts(pr["merged_at"]) if pr.get("merged_at") else None
    return any(
        state == "APPROVED"
        and is_human_reviewer(login)
        and base_login(login) != base_login(pr["author"])
        and (merged_at is None or not submitted_at or parse_ts(submitted_at) <= merged_at)
        for login, state, submitted_at in approvals
    )


def pr_ref(pr: dict) -> str:
    return f"{pr.get('repo')}#{pr.get('n')}"


def autonomous_window_stats(prs: list[dict], start: datetime, end: datetime) -> dict:
    """Autonomous share for PRs merged from start to end (UTC dates, inclusive), same as in_cycle."""
    window = [pr for pr in prs if counts_for_autonomous(pr) and in_cycle(pr["merged_at"], start, end)]
    repos = [r for r in REPOS if repo_active(r, end)] + sorted({pr.get("repo") for pr in window} - set(REPOS) - {None})
    by_repo = {r: {"labeled": 0, "total": 0} for r in repos}
    by_author = {g: {"labeled": 0, "total": 0, "unlabeled_unapproved": 0} for g in AUTONOMOUS_AUTHOR_GROUPS}
    labeled_unapproved, unlabeled_unapproved, approvals_unknown = [], [], []
    for pr in window:
        labeled = is_autonomous(pr)
        group = by_author[autonomous_author_group(pr["author"])]
        repo = by_repo.setdefault(pr.get("repo"), {"labeled": 0, "total": 0})
        for bucket in (group, repo):
            bucket["total"] += 1
            bucket["labeled"] += labeled
        approved = has_human_approval(pr)
        if approved is None:
            approvals_unknown.append(pr)
        elif not approved and labeled:
            labeled_unapproved.append(pr)
        elif not approved:
            unlabeled_unapproved.append(pr)
            group["unlabeled_unapproved"] += 1
    labeled_count = sum(1 for pr in window if is_autonomous(pr))
    return {
        "start": fmt_date(start),
        "end": fmt_date(end),
        "total": len(window),
        "labeled": labeled_count,
        "share": labeled_count / len(window) if window else None,
        "by_repo": by_repo,
        "by_author": by_author,
        "labeled_unapproved": labeled_unapproved,
        "unlabeled_unapproved": unlabeled_unapproved,
        "approvals_unknown": len(approvals_unknown),
    }


def build_autonomous_report(
    prs: list[dict],
    cycles: list[tuple[datetime, datetime]],
    collect_start: datetime,
    end_date: datetime,
) -> dict | None:
    """Autonomous-merge stats since tracking started, per cycle, and for the quarter to date.

    Returns None when the window does not reach back to the tracking start.
    """
    tracking_start = datetime.strptime(AUTONOMOUS_TRACKING_START, "%Y-%m-%d")
    if end_date < tracking_start or collect_start > tracking_start:
        return None
    quarter_start, _ = quarter_bounds(end_date)
    per_cycle = [
        {**autonomous_window_stats(prs, max(cs, tracking_start), ce), "cycle_start": fmt_date(cs), "cycle_end": fmt_date(ce)}
        for cs, ce in cycles
        if ce >= tracking_start
    ]
    return {
        "tracking_start": AUTONOMOUS_TRACKING_START,
        "since": autonomous_window_stats(prs, tracking_start, end_date),
        "quarter": autonomous_window_stats(prs, quarter_start, end_date) if collect_start <= quarter_start else None,
        "quarter_label": f"Q{(end_date.month - 1) // 3 + 1}'{end_date.strftime('%y')}",
        "per_cycle": per_cycle,
    }


def fmt_share(stats: dict) -> str:
    return f"{stats['share'] * 100:.1f}%" if stats["share"] is not None else "n/a"


def print_autonomous_report(report: dict) -> None:
    print()
    print("═" * 60)
    print(f"  AUTONOMOUS MERGES (label: {AUTONOMOUS_LABEL}; bots, Dependabot and Devin included)")
    print("═" * 60)
    windows = [("Since tracking start", report["since"])]
    if report["quarter"]:
        windows.append((f"{report['quarter_label']} to date", report["quarter"]))
    for name, stats in windows:
        print(f"  {name} ({stats['start']} → {stats['end']}): {stats['labeled']}/{stats['total']} = {fmt_share(stats)}")
        for group, g in stats["by_author"].items():
            print(f"    {group:<12} {g['labeled']:>3}/{g['total']:<4}")
        for repo, r in stats["by_repo"].items():
            print(f"    {str(repo):<22} {r['labeled']:>3}/{r['total']:<4}")
        print(f"    labeled, no human approval:   {len(stats['labeled_unapproved'])}")
        print(f"    unlabeled, no human approval: {len(stats['unlabeled_unapproved'])}")
        if stats["approvals_unknown"]:
            print(f"    ⚠️  approvals not collected for {stats['approvals_unknown']} PRs (left out of the cross-check)")
    print("  Per cycle:")
    for c in report["per_cycle"]:
        print(f"    {c['start']} → {c['end']}: {c['labeled']}/{c['total']} = {fmt_share(c)}")


# ── Main ────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    parser = argparse.ArgumentParser(description="Mindbloom PR metrics report")
    parser.add_argument("range", nargs="*", help="N cycles, or START END (YYYY-MM-DD)")
    parser.add_argument("--input", action="append", help="JSON file of pre-collected PRs (skips gh); repeatable")
    parser.add_argument("--history", help="JSON file of frozen per-cycle aggregates for earlier cycles")
    parser.add_argument(
        "--collect-from",
        help="YYYY-MM-DD: collect from this date even if --history covers it; frozen cycles from this date on "
             "only fill metrics the collected data lacks (opened, reviews)",
    )
    parser.add_argument("--as-of", help='Data retrieval time, e.g. "2026-10-21 ~9:31 AM CT" (marks the cutoff day as partial)')
    args = parser.parse_args()

    if len(args.range) == 1 and args.range[0].isdigit():
        num_cycles = int(args.range[0])
        end_date = today
        start_date = today - timedelta(days=num_cycles * CYCLE_DAYS)
    elif len(args.range) == 2:
        start_date = datetime.strptime(args.range[0], "%Y-%m-%d")
        end_date = datetime.strptime(args.range[1], "%Y-%m-%d")
    elif not args.range:
        end_date = today
        start_date = today - timedelta(days=CYCLE_DAYS)
    else:
        parser.error("expected N cycles or START END")
    start_date = start_date.replace(tzinfo=None)
    end_date = end_date.replace(tzinfo=None)

    history_cycles = []
    history_quarters = {}
    collect_start = start_date
    if args.history:
        with open(args.history) as f:
            history = json.load(f)
        history_cycles = history["cycles"]
        history_quarters = history.get("quarters", {})
        for cycle in history_cycles:
            if "reviews" not in cycle:
                cycle["reviews"] = sum(a["reviews"] for a in cycle["authors"].values())
        if history_cycles[0]["start"] != fmt_date(start_date):
            parser.error(f"START must be {history_cycles[0]['start']} (first frozen cycle) when --history is used")
        collect_start = datetime.strptime(history_cycles[-1]["end"], "%Y-%m-%d") + timedelta(days=1)
    history_fill: dict[tuple[str, str], dict] = {}
    if args.collect_from:
        collect_start = datetime.strptime(args.collect_from, "%Y-%m-%d")
        history_fill = {(c["start"], c["end"]): c for c in history_cycles if c["start"] >= args.collect_from}
        history_cycles = [c for c in history_cycles if c["end"] < args.collect_from]

    input_data = load_input_prs(args.input) if args.input else None
    opened_collected = input_data is None or input_data["opened"] is not None
    reviews_collected = input_data is None or input_data["reviews_collected"]
    ai_collected = input_data is None or input_data["ai_collected"]

    cycles = build_cycles(collect_start, end_date)

    print("═" * 60)
    print(f"  Mindbloom PR Metrics")
    print(f"  Period: {fmt_date(start_date)} → {fmt_date(end_date)} ({len(history_cycles) + len(cycles)} cycles)")
    if history_cycles:
        print(f"  Frozen history: {len(history_cycles)} cycles through {history_cycles[-1]['end']}")
    if input_data is not None:
        missing = [name for name, ok in (("opened", opened_collected), ("reviews", reviews_collected), ("AI PRs", ai_collected)) if not ok]
        print(f"  Input: {', '.join(args.input)} ({len(input_data['merged'])} merged PRs)")
        if missing:
            print(f"  Not collected in input: {', '.join(missing)}")
    print("═" * 60)

    # Store data per cycle
    all_cycle_merged: list[list[dict]] = []
    all_cycle_opened: list[list[dict]] = []

    for i, (cycle_start, cycle_end) in enumerate(cycles):
        label = f"Cycle {i + 1}: {fmt_date(cycle_start)} → {fmt_date(cycle_end)}"
        print()
        print("─" * 60)
        print(f"  {label}")
        print("─" * 60)

        cycle_merged = []
        cycle_opened = []
        if input_data is not None:
            cycle_merged = [
                pr for pr in input_data["merged"]
                if in_cycle(pr["merged_at"], cycle_start, cycle_end)
                and pr["merged_at"][:10] >= REPO_EFFECTIVE_START.get(pr.get("repo"), "")
            ]
            cycle_opened = [pr for pr in input_data["opened"] or [] if in_cycle(pr["created_at"], cycle_start, cycle_end)]
        else:
            for repo in REPOS:
                window = repo_window(repo, fmt_date(cycle_start), fmt_date(cycle_end))
                if window is None:
                    continue  # repo not effective yet: do not query it for this cycle
                s, e = window
                cycle_merged.extend(fetch_merged_prs(repo, s, e))
                cycle_opened.extend(fetch_opened_prs(repo, s, e))

        all_cycle_merged.append(cycle_merged)
        all_cycle_opened.append(cycle_opened)

        # Filter
        human_merged = [pr for pr in cycle_merged if is_human(pr["author"])]
        ai_merged = [pr for pr in cycle_merged if pr.get("is_ai_pr")]
        human_opened = [pr for pr in cycle_opened if is_human(pr["author"])]
        human_reviewers = [
            r for pr in cycle_merged for r in pr["reviewers"] if is_human(r)
        ]

        # Opened vs Merged
        opened_count = len(human_opened) if opened_collected else "n/a"
        merged_count = len(human_merged)
        ai_count = len(ai_merged) if ai_collected else "n/a"
        print()
        print(f"  Opened: {opened_count}  |  Merged: {merged_count}  |  AI PRs: {ai_count}")

        if not human_merged and not ai_merged:
            print("  No merged PRs this cycle.")
            continue

        # PRs Authored (merged, human only)
        if human_merged:
            print()
            print("  PRs Authored (merged):")
            for author, count in count_sorted([pr["author"] for pr in human_merged]):
                author_prs = [pr for pr in human_merged if pr["author"] == author]
                ttm = avg_ttm(author_prs)
                ttm_str = f"  avg merge: {fmt_duration(ttm)}" if ttm is not None else ""
                print(f"    {author:<25} {count:>3} PRs{ttm_str}")

        # AI PRs shepherded (attributed to approver)
        if ai_merged:
            print()
            print("  AI PRs Shepherded (by approver):")
            shepherds = [pr["shepherd"] for pr in ai_merged if pr.get("shepherd")]
            unattributed = sum(1 for pr in ai_merged if not pr.get("shepherd"))
            for shepherd, count in count_sorted(shepherds):
                print(f"    {shepherd:<25} {count:>3} PRs")
            if unattributed:
                print(f"    {'(no approver)' :<25} {unattributed:>3} PRs")

        if opened_collected:
            print()
            print("  PRs Opened:")
            for author, count in count_sorted([pr["author"] for pr in human_opened]):
                print(f"    {author:<25} {count:>3} PRs")

        if reviews_collected:
            print()
            print("  Reviews Given:")
            for reviewer, count in count_sorted(human_reviewers):
                print(f"    {reviewer:<25} {count:>3} reviews")

        # Cycle time-to-merge
        ttm_avg = avg_ttm(human_merged)
        ttm_med = median_ttm(human_merged)
        unique_authors = len(set(pr["author"] for pr in human_merged))
        print()
        print(f"  Cycle total: {merged_count} merged, {opened_count} opened, {ai_count} AI, {unique_authors} contributors")
        if ttm_avg is not None:
            print(f"  Time to merge: {fmt_duration(ttm_avg)} avg / {fmt_duration(ttm_med)} median")

    # ── Overall Summary ─────────────────────────────────────
    all_merged = [pr for cycle in all_cycle_merged for pr in cycle]
    all_opened = [pr for cycle in all_cycle_opened for pr in cycle]
    num_cycles = len(cycles)

    print()
    print("═" * 60)
    print(f"  OVERALL SUMMARY ({num_cycles} cycles)")
    print("═" * 60)

    human_merged = [pr for pr in all_merged if is_human(pr["author"])]
    ai_merged = [pr for pr in all_merged if pr.get("is_ai_pr")]
    human_opened = [pr for pr in all_opened if is_human(pr["author"])]
    human_reviewers = [r for pr in all_merged for r in pr["reviewers"] if is_human(r)]

    if not human_merged and not ai_merged:
        print("  No PRs found.")
        return

    print()
    print(
        f"  Total opened: {len(human_opened) if opened_collected else 'n/a'}  |  "
        f"Total merged: {len(human_merged)}  |  AI PRs: {len(ai_merged) if ai_collected else 'n/a'}"
    )

    print()
    print("  PRs Authored — merged (total | avg/cycle | avg merge time):")
    for author, count in count_sorted([pr["author"] for pr in human_merged]):
        avg = count / num_cycles
        author_prs = [pr for pr in human_merged if pr["author"] == author]
        ttm = avg_ttm(author_prs)
        ttm_str = fmt_duration(ttm) if ttm is not None else "n/a"
        print(f"    {author:<25} {count:>3} total   {avg:>5.1f}/cycle   merge: {ttm_str}")

    if ai_collected:
        print()
        print("  AI PRs Shepherded — by approver (total | avg/cycle):")
        shepherds_all = [pr["shepherd"] for pr in ai_merged if pr.get("shepherd")]
        unattributed_all = sum(1 for pr in ai_merged if not pr.get("shepherd"))
        for shepherd, count in count_sorted(shepherds_all):
            avg = count / num_cycles
            print(f"    {shepherd:<25} {count:>3} total   {avg:>5.1f}/cycle")
        if unattributed_all:
            print(f"    {'(no approver)':<25} {unattributed_all:>3} total")

    if opened_collected:
        print()
        print("  PRs Opened (total | avg/cycle):")
        for author, count in count_sorted([pr["author"] for pr in human_opened]):
            avg = count / num_cycles
            print(f"    {author:<25} {count:>3} total   {avg:>5.1f}/cycle")

    if reviews_collected:
        print()
        print("  Reviews Given (total | avg/cycle):")
        for reviewer, count in count_sorted(human_reviewers):
            avg = count / num_cycles
            print(f"    {reviewer:<25} {count:>3} total   {avg:>5.1f}/cycle")

    unique = len(set(pr["author"] for pr in human_merged))
    team_avg = len(human_merged) / unique / num_cycles
    ttm_avg = avg_ttm(human_merged)
    ttm_med = median_ttm(human_merged)

    print()
    print(f"  Contributors: {unique}")
    print(f"  Team avg PRs/person/cycle: {team_avg:.1f}")
    if ttm_avg is not None:
        print(f"  Team time to merge: {fmt_duration(ttm_avg)} avg / {fmt_duration(ttm_med)} median")

    # ── Trend ───────────────────────────────────────────────
    if num_cycles > 1:
        print()
        print("─" * 60)
        print("  CYCLE-OVER-CYCLE TREND")
        print("─" * 60)
        print(f"  {'Cycle':<28} {'Opened':>7} {'Merged':>7} {'Authors':>8} {'TTM avg':>8} {'TTM med':>8}")
        for i, (cycle_start, cycle_end) in enumerate(cycles):
            m = [pr for pr in all_cycle_merged[i] if is_human(pr["author"])]
            o = [pr for pr in all_cycle_opened[i] if is_human(pr["author"])]
            ca = len(set(pr["author"] for pr in m)) if m else 0
            t_avg = avg_ttm(m)
            t_med = median_ttm(m)
            t_avg_s = fmt_duration(t_avg) if t_avg else "-"
            t_med_s = fmt_duration(t_med) if t_med else "-"
            o_s = len(o) if opened_collected else "n/a"
            print(f"  {fmt_date(cycle_start)} → {fmt_date(cycle_end)}  {o_s:>7} {len(m):>7} {ca:>8} {t_avg_s:>8} {t_med_s:>8}")

    if input_data is not None:
        used = sum(len(c) for c in all_cycle_merged)
        if used != len(input_data["merged"]):
            print(f"  ⚠️  {len(input_data['merged']) - used} input PRs fall outside the report window and were ignored")

    # ── HTML Report ─────────────────────────────────────────
    collected_stats = [
        summarize_cycle(
            cs, ce, all_cycle_merged[i],
            all_cycle_opened[i] if opened_collected else None,
            reviews_collected, ai_collected,
        )
        for i, (cs, ce) in enumerate(cycles)
    ]
    for cycle in collected_stats:
        frozen = history_fill.get((cycle["start"], cycle["end"]))
        if frozen is None:
            continue
        for metric in ("opened", "reviews"):
            if cycle[metric] is not None:
                continue
            cycle[metric] = frozen[metric]
            cycle.setdefault("filled_from_history", []).append(metric)
            for author, stats in cycle["authors"].items():
                stats[metric] = 0
            for author, stats in frozen["authors"].items():
                cycle["authors"].setdefault(author, {
                    "merged": 0,
                    "opened": 0 if cycle["opened"] is not None else None,
                    "reviews": 0 if cycle["reviews"] is not None else None,
                    "shepherded": 0 if ai_collected else None,
                    "ttm_avg_hours": None,
                })[metric] = stats[metric]
    cycle_stats = history_cycles + collected_stats
    autonomous = build_autonomous_report(all_merged, cycles, collect_start, end_date)
    if autonomous:
        print_autonomous_report(autonomous)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "pr-metrics-report.html")
    generate_html_report(cycle_stats, history_quarters, start_date, end_date, report_path, args.as_of, autonomous)
    shutil.copyfile(report_path, os.path.join(script_dir, "index.html"))
    print(f"  📄 Copied to: {os.path.join(script_dir, 'index.html')}")


def generate_html_report(
    cycle_stats: list[dict],
    history_quarters: dict[str, dict],
    start_date: datetime,
    end_date: datetime,
    output_path: str = "pr-metrics-report.html",
    as_of: str | None = None,
    autonomous: dict | None = None,
):
    """Generate an interactive HTML report with Chart.js visualizations."""
    num_cycles = len(cycle_stats)
    cycle_labels = [f"{c['start']}→{c['end']}" for c in cycle_stats]

    def fmt_total(values: list) -> str:
        """Sum of collected values; '*' marks totals that miss uncollected cycles."""
        known = [v for v in values if v is not None]
        if not known:
            return "n/a"
        return f"{sum(known)}{'*' if len(known) < len(values) else ''}"

    cycle_total_key = {"merged": "merged", "opened": "opened", "shepherded": "ai", "reviews": "reviews"}

    def author_value(cycle: dict, author: str, key: str):
        """Per-author count for a cycle; None when the cycle did not collect that metric."""
        if cycle[cycle_total_key[key]] is None:
            return None
        return cycle["authors"].get(author, {}).get(key) or 0

    def author_ttm(cycle: dict, author: str) -> float | None:
        return cycle["authors"].get(author, {}).get("ttm_avg_hours")

    # Collect all human authors across all cycles
    all_authors = sorted(set(
        author
        for cycle in cycle_stats
        for author, stats in cycle["authors"].items()
        if stats["merged"] and is_human(author)
    ))

    # ── Build data structures for charts ────────────────────

    # Per-person merged PRs per cycle
    merged_by_author_cycle = {a: [] for a in all_authors}
    opened_by_author_cycle = {a: [] for a in all_authors}
    reviews_by_author_cycle = {a: [] for a in all_authors}
    ttm_by_author_cycle = {a: [] for a in all_authors}

    shepherd_by_author_cycle = {a: [] for a in all_authors}

    trend_opened = []
    trend_merged = []
    trend_ai = []
    trend_ttm_avg = []
    trend_ttm_med = []

    for cycle in cycle_stats:
        trend_opened.append(cycle["opened"])
        trend_merged.append(cycle["merged"])
        trend_ai.append(cycle["ai"])
        t_avg = cycle["ttm_avg_hours"]
        t_med = cycle["ttm_median_hours"]
        trend_ttm_avg.append(round(t_avg, 1) if t_avg else 0)
        trend_ttm_med.append(round(t_med, 1) if t_med else 0)

        for a in all_authors:
            merged_by_author_cycle[a].append(author_value(cycle, a, "merged"))
            opened_by_author_cycle[a].append(author_value(cycle, a, "opened"))
            reviews_by_author_cycle[a].append(author_value(cycle, a, "reviews"))
            shepherd_by_author_cycle[a].append(author_value(cycle, a, "shepherded"))
            ttm = author_ttm(cycle, a)
            ttm_by_author_cycle[a].append(round(ttm, 1) if ttm is not None else None)

    # Overall totals for summary cards
    total_human_merged = sum(c["merged"] for c in cycle_stats)
    total_human_opened = fmt_total([c["opened"] for c in cycle_stats])
    total_ai_merged = fmt_total([c["ai"] for c in cycle_stats])
    overall_ttm_avg = (
        sum(c["ttm_avg_hours"] * c["merged"] for c in cycle_stats if c["ttm_avg_hours"] is not None)
        / total_human_merged
        if total_human_merged else None
    )
    # Median needs per-PR values, which frozen history cycles do not keep.
    has_all_ttm_values = all("ttm_values" in c for c in cycle_stats)
    overall_ttm_med = (
        percentile([v for c in cycle_stats for v in c["ttm_values"]], 50) if has_all_ttm_values else None
    )
    not_collected = [
        name for name, key in (("opened", "opened"), ("AI PRs", "ai"), ("reviews", "reviews"))
        if any(c[key] is None for c in cycle_stats)
    ]
    uncollected_cycles = [f"{c['start']} → {c['end']}" for c in cycle_stats if c["opened"] is None or c["ai"] is None or c["reviews"] is None]
    collection_note = ""
    if not_collected:
        collection_note = (
            f"* Totals exclude {', '.join(uncollected_cycles)}: {', '.join(not_collected)} were not collected "
            f"for these cycles (pre-collected merged-PR input). Charts leave those points empty."
        )
    filled_cycles = [c for c in cycle_stats if c.get("filled_from_history")]
    if filled_cycles:
        filled_metrics = sorted({m for c in filled_cycles for m in c["filled_from_history"]})
        collection_note += (
            f" {' and '.join(filled_metrics).capitalize()} counts for {filled_cycles[0]['start']} → "
            f"{filled_cycles[-1]['end']} come from frozen history; all other metrics there use per-PR data."
        )
    if not has_all_ttm_values:
        collection_note += (
            " Median time to merge for the full window needs per-PR data that the frozen history"
            " does not keep; see per-quarter percentiles in the Time to Merge KPI section."
        )

    # ── Quarterly PRs per engineer ──────────────────────────
    def get_quarter(dt: datetime) -> str:
        q = (dt.month - 1) // 3 + 1
        return f"Q{q} {dt.year}"

    quarter_cycle_indices: dict[str, list[int]] = defaultdict(list)
    for i, cycle in enumerate(cycle_stats):
        quarter_cycle_indices[get_quarter(datetime.strptime(cycle["start"], "%Y-%m-%d"))].append(i)

    def merged_by_author(indices: list[int]) -> dict[str, int]:
        counts = defaultdict(int)
        for idx in indices:
            for author, stats in cycle_stats[idx]["authors"].items():
                if stats["merged"] and is_human(author):
                    counts[author] += stats["merged"]
        return counts

    def credited_by_author(indices: list[int]) -> dict[str, int]:
        """Devin PRs credited to their first human approver, in cycles where the rule applies."""
        counts = defaultdict(int)
        for idx in indices:
            if not cycle_stats[idx].get("devin_credit"):
                continue
            for author, stats in cycle_stats[idx]["authors"].items():
                if stats.get("shepherded") and is_human(author):
                    counts[author] += stats["shepherded"]
        return counts

    def kpi_by_author(indices: list[int]) -> dict[str, int]:
        counts = merged_by_author(indices)
        for author, credited in credited_by_author(indices).items():
            counts[author] += credited
        return counts

    quarterly_data = {q: kpi_by_author(idx) for q, idx in quarter_cycle_indices.items()}
    quarterly_own = {q: sum(merged_by_author(idx).values()) for q, idx in quarter_cycle_indices.items()}
    quarterly_credited = {q: sum(credited_by_author(idx).values()) for q, idx in quarter_cycle_indices.items()}
    quarterly_devin_credit = {
        q: all(cycle_stats[i].get("devin_credit") for i in idx) for q, idx in quarter_cycle_indices.items()
    }
    quarterly_devin_excluded = {
        q: sum(cycle_stats[i]["ai_unapproved"] for i in idx) if quarterly_devin_credit[q] else None
        for q, idx in quarter_cycle_indices.items()
    }

    # Build quarterly PRs/engineer data
    quarter_labels = sorted(quarterly_data.keys(), key=quarter_sort_key)
    uncredited_quarters = [q for q in quarter_labels if not quarterly_devin_credit[q]]
    devin_credit_note = (
        f"Devin crediting is not applied to {', '.join(uncredited_quarters)}: the frozen history has no "
        f"per-PR review data, so those quarters count human-authored PRs only."
        if uncredited_quarters else ""
    )
    quarterly_prs_per_eng = []
    quarterly_prs_per_eng_projection = []
    quarterly_total_merged = []
    quarterly_total_kpi = []
    quarterly_total_ai = []
    quarterly_engineer_count = []
    for q in quarter_labels:
        total = sum(quarterly_data[q].values())
        eng_count = len(quarterly_data[q])
        ai_values = [cycle_stats[i]["ai"] for i in quarter_cycle_indices[q] if cycle_stats[i]["ai"] is not None]
        quarterly_total_merged.append(quarterly_own[q])
        quarterly_total_kpi.append(total)
        quarterly_total_ai.append(sum(ai_values) if ai_values else None)
        quarterly_engineer_count.append(eng_count)
        quarterly_prs_per_eng.append(round(total / eng_count, 1) if eng_count else 0)
        quarterly_prs_per_eng_projection.append(None)

    # ── Time to merge percentiles per quarter ───────────────
    quarterly_ttm: dict[str, dict] = {}
    for q in quarter_labels:
        quarter_cycles = [cycle_stats[i] for i in quarter_cycle_indices[q]]
        if all("ttm_values" in c for c in quarter_cycles):
            values = [v for c in quarter_cycles for v in c["ttm_values"]]
            quarterly_ttm[q] = {"source": "computed"}
            for p in TTM_PERCENTILES:
                quarterly_ttm[q][p] = percentile(values, p)
        else:
            frozen = history_quarters.get(q, {})
            quarterly_ttm[q] = {"source": "frozen" if frozen else None, "note": frozen.get("ttm_note")}
            for p in TTM_PERCENTILES:
                quarterly_ttm[q][p] = frozen.get(f"ttm_p{p}_hours")

    current_quarter_label = get_quarter(end_date)
    current_quarter_start, current_quarter_end = quarter_bounds(end_date)
    current_quarter_total_days = (current_quarter_end - current_quarter_start).days + 1
    current_quarter_elapsed_days = max(1, min((end_date - current_quarter_start).days + 1, current_quarter_total_days))
    current_quarter_target = QUARTERLY_PRS_PER_ENGINEER_TARGETS.get(current_quarter_label)
    current_quarter_total_merged = sum(quarterly_data.get(current_quarter_label, {}).values())
    current_quarter_engineer_count = len(quarterly_data.get(current_quarter_label, {}))
    current_quarter_prs_per_eng = (
        round(current_quarter_total_merged / current_quarter_engineer_count, 1)
        if current_quarter_engineer_count else 0
    )
    current_quarter_projected_prs_per_eng = (
        round(current_quarter_prs_per_eng * current_quarter_total_days / current_quarter_elapsed_days, 1)
        if current_quarter_engineer_count else 0
    )
    current_quarter_days_remaining = max(current_quarter_total_days - current_quarter_elapsed_days, 0)
    current_quarter_progress_pct = round(current_quarter_elapsed_days / current_quarter_total_days * 100, 1)
    current_quarter_partial = end_date < current_quarter_end
    current_quarter_heading = f"{current_quarter_label}{' to date (partial quarter)' if current_quarter_partial else ''}"
    data_through = f"Data through {fmt_date(end_date)}"
    if as_of:
        data_through += f", retrieved {as_of}; the cutoff day is partial"
    if current_quarter_target is None:
        current_quarter_status = "No target"
        current_quarter_status_class = ""
        current_quarter_summary = (
            f"At the current pace, {current_quarter_label} is projected to finish at "
            f"{current_quarter_projected_prs_per_eng:.1f} PRs per engineer. No KPI target is set for this quarter."
        )
        current_quarter_target_text = f"No KPI target set for {current_quarter_label}"
    else:
        current_quarter_target_total_prs = current_quarter_target * current_quarter_engineer_count
        current_quarter_prs_needed = max(current_quarter_target_total_prs - current_quarter_total_merged, 0)
        current_quarter_required_daily_prs = (
            round(current_quarter_prs_needed / current_quarter_days_remaining, 1)
            if current_quarter_days_remaining else 0
        )
        current_quarter_on_track = current_quarter_projected_prs_per_eng >= current_quarter_target
        current_quarter_status = "On track" if current_quarter_on_track else "Off track"
        current_quarter_status_class = "kpi-on-track" if current_quarter_on_track else "kpi-off-track"
        current_quarter_gap = round(current_quarter_projected_prs_per_eng - current_quarter_target, 1)
        current_quarter_summary = (
            f"At the current pace, {current_quarter_label} is projected to finish at "
            f"{current_quarter_projected_prs_per_eng:.1f} PRs per engineer, "
            f"{abs(current_quarter_gap):.1f} {'above' if current_quarter_on_track else 'below'} "
            f"the {current_quarter_target} PR/engineer KPI."
        )
        if not current_quarter_on_track and current_quarter_days_remaining:
            current_quarter_summary += (
                f" Need {current_quarter_prs_needed} more credited PRs across the team, "
                f"or about {current_quarter_required_daily_prs:.1f}/day for the rest of the quarter."
            )
        current_quarter_target_text = f"KPI target: {current_quarter_target} PRs per engineer for {current_quarter_label}"

    if current_quarter_label in quarter_labels:
        current_quarter_index = quarter_labels.index(current_quarter_label)
        quarterly_prs_per_eng_projection[current_quarter_index] = current_quarter_projected_prs_per_eng

    quarterly_target_line = [QUARTERLY_PRS_PER_ENGINEER_TARGETS.get(q) for q in quarter_labels]

    current_quarter_own = quarterly_own.get(current_quarter_label, 0)
    current_quarter_credited = quarterly_credited.get(current_quarter_label, 0)
    current_quarter_excluded = quarterly_devin_excluded.get(current_quarter_label)
    excluded_users = ", ".join(sorted(EXCLUDED_USERS))
    if quarterly_devin_credit.get(current_quarter_label):
        kpi_definition = (
            "PRs/engineer = (human-authored merged PRs + merged Devin PRs credited to their first human approver) "
            "÷ engineers with at least one credited PR this quarter; Devin PRs with no human approval are excluded."
        )
    else:
        kpi_definition = "PRs/engineer = human merged PRs ÷ engineers with at least one merged PR this quarter."
    kpi_definition += f" Bots and {excluded_users} are excluded. Projection is straight-line from days elapsed."

    # ── Time to merge KPI (current quarter) ─────────────────
    def fmt_ttm(hours: float | None) -> str:
        return fmt_duration(hours) if hours is not None else "n/a"

    def fmt_ttm_detail(hours: float | None) -> str:
        if hours is None or hours < 24:
            return fmt_ttm(hours)
        return f"{fmt_duration(hours)} ({hours:.1f}h)"

    def ttm_card_label(name: str, hours: float | None) -> str:
        return f"{name} Time to Merge{f' · {hours:.1f}h' if hours is not None and hours >= 24 else ''}"

    def fmt_goal_days(hours: float) -> str:
        return f"{hours / 24:g} days"

    current_ttm = quarterly_ttm.get(current_quarter_label, {p: None for p in TTM_PERCENTILES})
    current_ttm_target = TTM_P90_TARGET_HOURS.get(current_quarter_label)
    current_p90 = current_ttm[90]
    if current_ttm_target is None or current_p90 is None:
        ttm_status, ttm_status_class = "No target", ""
    elif current_p90 <= current_ttm_target:
        ttm_status, ttm_status_class = "On track", "kpi-on-track"
    else:
        ttm_status, ttm_status_class = "Off track", "kpi-off-track"
    ttm_summary = f"P90 time to merge is {fmt_ttm_detail(current_p90)}"
    if current_p90 is not None:
        if current_ttm_target is not None:
            gap = current_p90 - current_ttm_target
            ttm_summary += f", {fmt_duration(abs(gap))} {'within' if gap <= 0 else 'above'} the {current_ttm_target / 24:g}-day goal"
    ttm_summary += "."
    previous_quarters = [q for q in quarter_labels if quarter_sort_key(q) < quarter_sort_key(current_quarter_label) and quarterly_ttm[q][90] is not None]
    if previous_quarters:
        prev = previous_quarters[-1]
        ttm_summary += f" {prev} baseline: P90 {fmt_ttm(quarterly_ttm[prev][90])}, P50 {fmt_ttm(quarterly_ttm[prev][50])}."

    ttm_rows = ""
    for q in quarter_labels:
        stats = quarterly_ttm[q]
        goal = TTM_P90_TARGET_HOURS.get(q)
        if goal is None or stats[90] is None:
            row_status = "Baseline" if stats[90] is not None else "Not measured"
            status_html = row_status
        else:
            ok = stats[90] <= goal
            status_html = f'<span class="kpi-status {"kpi-on-track" if ok else "kpi-off-track"}">{"On track" if ok else "Off track"}</span>'
        partial = " (to date)" if q == current_quarter_label and current_quarter_partial else ""
        ttm_rows += f"""      <tr>
        <td>{q}{partial}</td>
        <td class="num">{quarterly_own[q]}</td>
        <td class="num">{fmt_ttm_detail(stats[50])}</td>
        <td class="num">{fmt_ttm_detail(stats[75])}</td>
        <td class="num"><strong>{fmt_ttm_detail(stats[90])}</strong></td>
        <td class="num">{fmt_ttm_detail(stats[95])}</td>
        <td class="num">{f"≤ {fmt_goal_days(goal)}" if goal is not None else "—"}</td>
        <td class="num">{status_html}</td>
      </tr>\n"""
    ttm_notes = " ".join(
        f"{q}: {quarterly_ttm[q]['note']}" for q in quarter_labels if quarterly_ttm[q].get("note")
    )
    unmeasured = [q for q in quarter_labels if quarterly_ttm[q]["source"] is None]
    if unmeasured:
        ttm_notes += f" {', '.join(unmeasured)}: percentiles were not recorded and the frozen history has no per-PR data."

    # Colors for each author
    palette = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
        "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
        "#9c755f", "#bab0ac",
    ]
    author_colors = {a: palette[i % len(palette)] for i, a in enumerate(all_authors)}

    def js_datasets(data_by_author, label_suffix=""):
        """Build Chart.js dataset array."""
        datasets = []
        for a in all_authors:
            datasets.append({
                "label": a,
                "data": data_by_author[a],
                "backgroundColor": author_colors[a],
                "borderColor": author_colors[a],
                "borderWidth": 2,
            })
        return json.dumps(datasets)

    def summarize_authors(indices: list[int], frozen_avg_merge_time: dict[str, str] | None = None) -> list[dict]:
        """Build per-author summary rows for a set of cycles.

        frozen_avg_merge_time keeps closed-quarter values exact, since frozen
        per-cycle averages are rounded to 0.1h.
        """
        cycles_in_window = [cycle_stats[i] for i in indices]
        merged_counts = merged_by_author(indices)
        credited_counts = credited_by_author(indices)
        kpi_counts = kpi_by_author(indices)
        authors = sorted(
            kpi_counts.keys(),
            key=lambda author: (-kpi_counts[author], -merged_counts.get(author, 0), author.lower()),
        )

        rows = []
        for author in authors:
            ttm_hours = ttm_count = 0
            for cycle in cycles_in_window:
                stats = cycle["authors"].get(author)
                if stats and stats["merged"] and stats["ttm_avg_hours"] is not None:
                    ttm_hours += stats["ttm_avg_hours"] * stats["merged"]
                    ttm_count += stats["merged"]
            rows.append({
                "author": author,
                "opened": fmt_total([author_value(c, author, "opened") for c in cycles_in_window]),
                "merged": merged_counts.get(author, 0),
                "shepherded": fmt_total([author_value(c, author, "shepherded") for c in cycles_in_window]),
                "credited": credited_counts.get(author, 0),
                "kpi_total": kpi_counts[author],
                "reviews": fmt_total([author_value(c, author, "reviews") for c in cycles_in_window]),
                "avg_merge_time": (frozen_avg_merge_time or {}).get(author)
                or (fmt_duration(ttm_hours / ttm_count) if ttm_count else "n/a"),
                "merged_per_cycle": merged_counts.get(author, 0) / len(indices) if indices else 0,
            })
        return rows

    def render_summary_table(title: str, rows: list[dict], subtitle: str | None = None, credited: bool = False) -> str:
        """Render a summary table section.

        credited tables show own PRs, Devin PRs credited as first human approver, and the KPI total.
        """
        subtitle_html = f'\n  <p class="section-subtitle">{subtitle}</p>' if subtitle else ""
        if credited:
            pr_headers = """        <th class="num">Own Merged</th>
        <th class="num">Devin Credited</th>
        <th class="num">KPI Total</th>"""
        else:
            pr_headers = """        <th class="num">Merged</th>
        <th class="num">AI Shepherded</th>"""
        html_table = f"""<div class="chart-box summary-box">
  <h2>{title}</h2>{subtitle_html}
  <table>
    <thead>
      <tr>
        <th>Author</th>
        <th class="num">Opened</th>
{pr_headers}
        <th class="num">Reviews</th>
        <th class="num">Avg Merge Time</th>
        <th class="num">Merged/Cycle</th>
      </tr>
    </thead>
    <tbody>
"""
        for row in rows:
            if credited:
                pr_cells = f"""        <td class="num">{row["merged"]}</td>
        <td class="num">{row["credited"]}</td>
        <td class="num"><strong>{row["kpi_total"]}</strong></td>"""
            else:
                pr_cells = f"""        <td class="num">{row["merged"]}</td>
        <td class="num">{row["shepherded"]}</td>"""
            html_table += f"""      <tr>
        <td>{row["author"]}</td>
        <td class="num">{row["opened"]}</td>
{pr_cells}
        <td class="num">{row["reviews"]}</td>
        <td class="num">{row["avg_merge_time"]}</td>
        <td class="num">{row["merged_per_cycle"]:.1f}</td>
      </tr>\n"""
        html_table += """    </tbody>
  </table>
</div>
"""
        return html_table

    overall_summary_rows = summarize_authors(list(range(num_cycles)))
    overall_credited = any(c.get("devin_credit") for c in cycle_stats)
    overall_subtitle = (
        f"KPI Total = own merged PRs + Devin PRs credited to the first human approver. {devin_credit_note}"
        if overall_credited else None
    )

    quarterly_summary_sections = []
    for quarter in quarter_labels:
        cycle_indices = quarter_cycle_indices[quarter]
        quarter_cycles = [cycle_stats[i] for i in cycle_indices]
        frozen_quarter = not any("ttm_values" in c for c in quarter_cycles)
        summary_rows = summarize_authors(
            cycle_indices,
            history_quarters.get(quarter, {}).get("author_avg_merge_time") if frozen_quarter else None,
        )
        eng_count = len(quarterly_data[quarter])
        kpi_total = sum(quarterly_data[quarter].values())
        ppe = round(kpi_total / eng_count, 1) if eng_count else 0
        if quarterly_devin_credit[quarter]:
            subtitle = (
                f"{fmt_total([c['opened'] for c in quarter_cycles])} opened · "
                f"{quarterly_own[quarter]} human merged + {quarterly_credited[quarter]} Devin credited = "
                f"{kpi_total} KPI PRs · {eng_count} engineers · "
                f"{fmt_total([c['ai'] for c in quarter_cycles])} Devin PRs merged, "
                f"{quarterly_devin_excluded[quarter]} excluded (no human approval)"
            )
        else:
            subtitle = (
                f"{fmt_total([c['opened'] for c in quarter_cycles])} opened · "
                f"{kpi_total} merged · "
                f"{fmt_total([c['ai'] for c in quarter_cycles])} AI PRs · "
                f"{eng_count} contributors · Devin crediting not applied (no per-PR review data)"
            )
        goal = QUARTERLY_PRS_PER_ENGINEER_TARGETS.get(quarter)
        if quarter == current_quarter_label and current_quarter_partial:
            subtitle += f" · partial quarter through {fmt_date(end_date)}"
        elif goal is not None:
            subtitle += f" · {ppe:.1f} PRs/engineer vs {goal} goal ({'Achieved' if ppe >= goal else 'Missed'})"
        quarterly_summary_sections.append(
            render_summary_table(quarter, summary_rows, subtitle, credited=quarterly_devin_credit[quarter])
        )

    # ── Autonomous merges section ───────────────────────────
    autonomous_html = autonomous_js = ""
    if autonomous:
        since, quarter = autonomous["since"], autonomous["quarter"]

        def share_row(name: str, stats: dict) -> str:
            return f"""      <tr>
        <td>{name}</td>
        <td class="num">{stats["labeled"]}</td>
        <td class="num">{stats["total"]}</td>
        <td class="num"><strong>{fmt_share(stats)}</strong></td>
      </tr>\n"""

        repo_rows = "".join(share_row(repo, r | {"share": r["labeled"] / r["total"] if r["total"] else None})
                            for repo, r in since["by_repo"].items())
        repo_rows += share_row("All repos", since)
        author_rows = ""
        for group, g in since["by_author"].items():
            author_rows += f"""      <tr>
        <td>{group}</td>
        <td class="num">{g["labeled"]}</td>
        <td class="num">{g["total"]}</td>
        <td class="num"><strong>{fmt_share(g | {"share": g["labeled"] / g["total"] if g["total"] else None})}</strong></td>
        <td class="num">{g["unlabeled_unapproved"]}</td>
      </tr>\n"""
        author_rows += f"""      <tr>
        <td>All authors</td>
        <td class="num">{since["labeled"]}</td>
        <td class="num">{since["total"]}</td>
        <td class="num"><strong>{fmt_share(since)}</strong></td>
        <td class="num">{len(since["unlabeled_unapproved"])}</td>
      </tr>\n"""
        quarter_cards = ""
        if quarter:
            quarter_cards = f"""
    <div class="card"><div class="value">{fmt_share(quarter)}</div><div class="label">{autonomous["quarter_label"]} To Date Share</div></div>
    <div class="card"><div class="value">{quarter["labeled"]} / {quarter["total"]}</div><div class="label">{autonomous["quarter_label"]} To Date Labeled / Merged</div></div>"""
        labeled_unapproved_refs = (
            ", ".join(pr_ref(pr) for pr in since["labeled_unapproved"]) if len(since["labeled_unapproved"]) <= 10 else ""
        )
        unknown_note = (
            f" Approvals were not collected for {since['approvals_unknown']} PRs; they are left out of both cross-check counts."
            if since["approvals_unknown"] else ""
        )
        autonomous_html = f"""<div class="chart-box kpi-box">
  <h2>Autonomous merges</h2>
  <p class="section-subtitle">Share of merged PRs carrying the <code>{AUTONOMOUS_LABEL}</code> label. Tracking started {autonomous["tracking_start"]}. {data_through}. No goal is set for Q4.</p>
  <div class="cards kpi-grid">
    <div class="card"><div class="value">{fmt_share(since)}</div><div class="label">Share Since {autonomous["tracking_start"]}</div></div>
    <div class="card"><div class="value">{since["labeled"]} / {since["total"]}</div><div class="label">Labeled / Merged Since {autonomous["tracking_start"]}</div></div>{quarter_cards}
    <div class="card"><div class="value">{len(since["labeled_unapproved"])}</div><div class="label">Labeled, No Human Approval (a)</div></div>
    <div class="card"><div class="value">{len(since["unlabeled_unapproved"])}</div><div class="label">Unlabeled, No Human Approval (b)</div></div>
  </div>
  <div class="chart-grid">
    <div class="chart-box">
      <h2>Autonomous share (per cycle)</h2>
      <canvas id="autonomousChart"></canvas>
    </div>
    <div class="chart-box">
      <h2>By repo (since {autonomous["tracking_start"]})</h2>
      <table>
        <thead>
          <tr><th>Repo</th><th class="num">Labeled</th><th class="num">Total</th><th class="num">Share</th></tr>
        </thead>
        <tbody>
{repo_rows}        </tbody>
      </table>
      <h2 style="margin-top: 24px;">By author (since {autonomous["tracking_start"]})</h2>
      <table>
        <thead>
          <tr><th>Author</th><th class="num">Labeled</th><th class="num">Total</th><th class="num">Share</th><th class="num">Unlabeled, No Human Approval</th></tr>
        </thead>
        <tbody>
{author_rows}        </tbody>
      </table>
    </div>
  </div>
  <p class="footnote"><strong>Definition:</strong> autonomous share = merged PRs labeled <code>{AUTONOMOUS_LABEL}</code> ÷ all merged PRs in the configured repos ({", ".join(REPOS)}) in the same window. Unlike the PRs/engineer and time-to-merge KPIs, Dependabot, other bot and Devin PRs count in both numerator and denominator; {excluded_users} is excluded. Dates are UTC merge dates. The label is read when the report is generated. Cycles are the dashboard's 2-week cycles, starting at the tracking date.</p>
  <p class="footnote"><strong>Cross-check:</strong> (a) = labeled PRs merged with no human APPROVED review (of {since["labeled"]} labeled); (b) = unlabeled PRs merged with no human APPROVED review, a sign of possible under-labeling (for example queued PRs). Bots, Devin and the merge queue app are not human reviewers. Only approvals submitted before the merge count.{unknown_note}{f" (a): {labeled_unapproved_refs}." if labeled_unapproved_refs else ""}{f" Q4 to date also counts merges from {quarter['start']}, before the label existed, as unlabeled." if quarter else ""}</p>
</div>
"""
        autonomous_js = f"""
// Autonomous share per cycle (no goal line: no goal is set for Q4)
new Chart(document.getElementById('autonomousChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps([f"{c['start']}→{c['end']}" for c in autonomous["per_cycle"]])},
    datasets: [
      {{ label: 'Autonomous share (%)', data: {json.dumps([round(c["share"] * 100, 1) if c["share"] is not None else None for c in autonomous["per_cycle"]])}, backgroundColor: '#58a6ff', borderColor: '#58a6ff', borderWidth: 2 }},
    ],
  }},
  options: {{
    ...chartDefaults,
    scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, max: 100 }} }},
    plugins: {{
      ...chartDefaults.plugins,
      tooltip: {{
        callbacks: {{
          afterLabel: function(ctx) {{
            const labeled = {json.dumps([c["labeled"] for c in autonomous["per_cycle"]])};
            const totals = {json.dumps([c["total"] for c in autonomous["per_cycle"]])};
            return `${{labeled[ctx.dataIndex]}} labeled / ${{totals[ctx.dataIndex]}} merged`;
          }}
        }}
      }}
    }}
  }},
}});
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mindbloom PR Metrics — {fmt_date(start_date)} to {fmt_date(end_date)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f1117; color: #e1e4e8; padding: 24px; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 4px; color: #fff; }}
  .subtitle {{ color: #8b949e; margin-bottom: 24px; font-size: 0.95rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; text-align: center; }}
  .card .value {{ font-size: 2rem; font-weight: 700; color: #58a6ff; }}
  .card .label {{ font-size: 0.8rem; color: #8b949e; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
  .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 24px; margin-bottom: 32px; }}
  .chart-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }}
  .chart-box h2 {{ font-size: 1.1rem; margin-bottom: 16px; color: #c9d1d9; }}
  canvas {{ max-height: 350px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #21262d; font-size: 0.9rem; }}
  th {{ color: #8b949e; font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px; }}
  td {{ color: #c9d1d9; }}
  tr:hover td {{ background: #1c2129; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .summary-grid {{ display: grid; gap: 24px; margin-bottom: 32px; }}
  .summary-box {{ margin-bottom: 0; }}
  .section-subtitle {{ color: #8b949e; margin: -8px 0 16px; font-size: 0.9rem; }}
  .kpi-grid {{ margin-bottom: 16px; }}
  .kpi-box {{ margin-bottom: 32px; }}
  .kpi-summary {{ color: #c9d1d9; line-height: 1.5; }}
  .kpi-summary strong {{ color: #fff; }}
  .kpi-status {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-on-track {{ color: #3fb950; background: rgba(63, 185, 80, 0.14); }}
  .kpi-off-track {{ color: #ff7b72; background: rgba(255, 123, 114, 0.14); }}
  .footnote {{ color: #8b949e; font-size: 0.85rem; line-height: 1.5; margin: -16px 0 32px; }}
  .kpi-box .footnote {{ margin: 12px 0 0; }}
  code {{ background: #21262d; border-radius: 4px; padding: 1px 5px; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>📊 Mindbloom PR Metrics</h1>
<p class="subtitle">{fmt_date(start_date)} → {fmt_date(end_date)} · {num_cycles} cycle{"s" if num_cycles != 1 else ""} · {len(all_authors)} contributors</p>

<div class="cards">
  <div class="card"><div class="value">{total_human_opened}</div><div class="label">PRs Opened</div></div>
  <div class="card"><div class="value">{total_human_merged}</div><div class="label">PRs Merged (Human)</div></div>
  <div class="card"><div class="value">{total_ai_merged}</div><div class="label">AI PRs Merged</div></div>
  <div class="card"><div class="value">{len(all_authors)}</div><div class="label">Contributors</div></div>
  <div class="card"><div class="value">{fmt_duration(overall_ttm_avg) if overall_ttm_avg else 'n/a'}</div><div class="label">Avg Time to Merge</div></div>
  <div class="card"><div class="value">{fmt_duration(overall_ttm_med) if overall_ttm_med else ('n/a*' if not has_all_ttm_values else 'n/a')}</div><div class="label">Median Time to Merge</div></div>
  <div class="card"><div class="value">{total_human_merged / len(all_authors) / num_cycles:.1f}</div><div class="label">Avg PRs/Person/Cycle</div></div>
</div>
{f'<p class="footnote">{collection_note.strip()}</p>' if collection_note else ''}

<div class="chart-grid">
  <div class="chart-box">
    <h2>Opened vs Merged (per cycle)</h2>
    <canvas id="trendChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Time to Merge (per cycle)</h2>
    <canvas id="ttmTrendChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>PRs Merged by Author (per cycle)</h2>
    <canvas id="mergedChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>PRs Opened by Author (per cycle)</h2>
    <canvas id="openedChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>AI PRs Shepherded by Approver (per cycle)</h2>
    <canvas id="shepherdChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Reviews Given by Author (per cycle)</h2>
    <canvas id="reviewsChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Avg Time to Merge by Author (hours)</h2>
    <canvas id="ttmAuthorChart"></canvas>
  </div>
</div>

<div class="chart-box kpi-box">
  <h2>{current_quarter_heading} KPI Pace</h2>
  <p class="section-subtitle">{current_quarter_target_text} ({fmt_date(current_quarter_start)} → {fmt_date(current_quarter_end)}). {data_through}. {kpi_definition}</p>
  <div class="cards kpi-grid">
    <div class="card"><div class="value">{current_quarter_prs_per_eng:.1f}</div><div class="label">Actual PRs/Engineer</div></div>
    <div class="card"><div class="value">{current_quarter_projected_prs_per_eng:.1f}</div><div class="label">Projected PRs/Engineer</div></div>
    <div class="card"><div class="value">{current_quarter_total_merged}</div><div class="label">KPI PRs This Quarter{f" ({current_quarter_own} human + {current_quarter_credited} Devin)" if current_quarter_excluded is not None else ""}</div></div>
    <div class="card"><div class="value">{current_quarter_engineer_count}</div><div class="label">Active Engineers</div></div>
    <div class="card"><div class="value">{current_quarter_progress_pct:.1f}%</div><div class="label">Quarter Elapsed</div></div>
    {f'<div class="card"><div class="value">{current_quarter_excluded}</div><div class="label">Devin PRs Excluded (no human approval; info only)</div></div>' if current_quarter_excluded is not None else ""}
    <div class="card"><div class="value"><span class="kpi-status {current_quarter_status_class}">{current_quarter_status}</span></div><div class="label">KPI Status</div></div>
  </div>
  <p class="kpi-summary"><strong>{current_quarter_label} outlook:</strong> {current_quarter_summary}</p>
  {f'<p class="footnote">{devin_credit_note}</p>' if devin_credit_note else ''}
  <p class="footnote">{REPOS_NOTE}</p>
</div>

<div class="chart-box kpi-box">
  <h2>{current_quarter_heading} Time to Merge KPI</h2>
  <p class="section-subtitle">Headline: P90 time from PR created to merged for human-authored merged PRs (Devin PRs are not included){f"; goal P90 ≤ {fmt_goal_days(current_ttm_target)}" if current_ttm_target is not None else ""}. P50 is context. Percentiles use linear interpolation. {data_through}.</p>
  <div class="cards kpi-grid">
    <div class="card"><div class="value">{fmt_ttm(current_ttm[90])}</div><div class="label">{ttm_card_label("P90", current_ttm[90])}</div></div>
    <div class="card"><div class="value">{fmt_ttm(current_ttm[50])}</div><div class="label">{ttm_card_label("P50", current_ttm[50])}</div></div>
    <div class="card"><div class="value">{fmt_ttm(current_ttm[75])}</div><div class="label">{ttm_card_label("P75", current_ttm[75])}</div></div>
    <div class="card"><div class="value">{fmt_ttm(current_ttm[95])}</div><div class="label">{ttm_card_label("P95", current_ttm[95])}</div></div>
    <div class="card"><div class="value"><span class="kpi-status {ttm_status_class}">{ttm_status}</span></div><div class="label">P90 KPI Status</div></div>
  </div>
  <p class="kpi-summary"><strong>{current_quarter_label} time to merge:</strong> {ttm_summary}</p>
  <table>
    <thead>
      <tr>
        <th>Quarter</th>
        <th class="num">Human Merged PRs</th>
        <th class="num">P50</th>
        <th class="num">P75</th>
        <th class="num">P90</th>
        <th class="num">P95</th>
        <th class="num">P90 Goal</th>
        <th class="num">Status</th>
      </tr>
    </thead>
    <tbody>
{ttm_rows}    </tbody>
  </table>
  {f'<p class="footnote">{ttm_notes.strip()}</p>' if ttm_notes.strip() else ''}
</div>

{autonomous_html}
<div class="chart-grid">
  <div class="chart-box">
    <h2>KPI PRs per Engineer (Quarterly)</h2>
    <canvas id="qtrPerEngChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Quarterly Totals (Merged + AI)</h2>
    <canvas id="qtrTotalsChart"></canvas>
  </div>
</div>

{render_summary_table("Overall Summary", overall_summary_rows, overall_subtitle, credited=overall_credited)}

<div class="summary-grid">
  <div class="chart-box summary-box">
    <h2>Quarterly Summary</h2>
    <p class="section-subtitle">Per-author rollups for each quarter in this reporting window.</p>
  </div>
{''.join(quarterly_summary_sections)}
</div>

<script>
const labels = {json.dumps(cycle_labels)};
const chartDefaults = {{
  responsive: true,
  plugins: {{
    legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
  }},
  scales: {{
    x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }}, beginAtZero: true }},
  }},
}};

// Opened vs Merged trend
new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Opened', data: {json.dumps(trend_opened)}, borderColor: '#f28e2b', backgroundColor: 'rgba(242,142,43,0.1)', fill: true, tension: 0.3 }},
      {{ label: 'Merged (Human)', data: {json.dumps(trend_merged)}, borderColor: '#4e79a7', backgroundColor: 'rgba(78,121,167,0.1)', fill: true, tension: 0.3 }},
      {{ label: 'AI PRs', data: {json.dumps(trend_ai)}, borderColor: '#b07aa1', backgroundColor: 'rgba(176,122,161,0.1)', fill: true, tension: 0.3, borderDash: [5, 5] }},
    ],
  }},
  options: chartDefaults,
}});

// TTM trend
new Chart(document.getElementById('ttmTrendChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Avg (hours)', data: {json.dumps(trend_ttm_avg)}, borderColor: '#e15759', tension: 0.3 }},
      {{ label: 'Median (hours)', data: {json.dumps(trend_ttm_med)}, borderColor: '#76b7b2', tension: 0.3 }},
    ],
  }},
  options: chartDefaults,
}});

// Stacked bar: merged by author
new Chart(document.getElementById('mergedChart'), {{
  type: 'bar',
  data: {{ labels, datasets: {js_datasets(merged_by_author_cycle)} }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, x: {{ ...chartDefaults.scales.x, stacked: true }}, y: {{ ...chartDefaults.scales.y, stacked: true }} }} }},
}});

// Stacked bar: opened by author
new Chart(document.getElementById('openedChart'), {{
  type: 'bar',
  data: {{ labels, datasets: {js_datasets(opened_by_author_cycle)} }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, x: {{ ...chartDefaults.scales.x, stacked: true }}, y: {{ ...chartDefaults.scales.y, stacked: true }} }} }},
}});

// Stacked bar: AI PRs shepherded
new Chart(document.getElementById('shepherdChart'), {{
  type: 'bar',
  data: {{ labels, datasets: {js_datasets(shepherd_by_author_cycle)} }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, x: {{ ...chartDefaults.scales.x, stacked: true }}, y: {{ ...chartDefaults.scales.y, stacked: true }} }} }},
}});

// Stacked bar: reviews by author
new Chart(document.getElementById('reviewsChart'), {{
  type: 'bar',
  data: {{ labels, datasets: {js_datasets(reviews_by_author_cycle)} }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, x: {{ ...chartDefaults.scales.x, stacked: true }}, y: {{ ...chartDefaults.scales.y, stacked: true }} }} }},
}});

// TTM by author (grouped bar)
new Chart(document.getElementById('ttmAuthorChart'), {{
  type: 'bar',
  data: {{ labels, datasets: {js_datasets(ttm_by_author_cycle)} }},
  options: chartDefaults,
}});

// Quarterly PRs per engineer
const qtrLabels = {json.dumps(quarter_labels)};
new Chart(document.getElementById('qtrPerEngChart'), {{
  type: 'bar',
  data: {{
    labels: qtrLabels,
    datasets: [
      {{ label: 'Actual PRs/Engineer', data: {json.dumps(quarterly_prs_per_eng)}, backgroundColor: '#58a6ff', borderColor: '#58a6ff', borderWidth: 2 }},
      {{ label: 'Projected Quarter Pace', data: {json.dumps(quarterly_prs_per_eng_projection)}, backgroundColor: '#f28e2b', borderColor: '#f28e2b', borderWidth: 2 }},
      {{ label: 'KPI Target', data: {json.dumps(quarterly_target_line)}, type: 'line', borderColor: '#3fb950', backgroundColor: '#3fb950', borderWidth: 2, borderDash: [6, 6], pointRadius: 0, tension: 0 }},
    ],
  }},
  options: {{
    ...chartDefaults,
    plugins: {{
      ...chartDefaults.plugins,
      tooltip: {{
        callbacks: {{
          afterLabel: function(ctx) {{
            const i = ctx.dataIndex;
            const counts = {json.dumps(quarterly_engineer_count)};
            const totals = {json.dumps(quarterly_total_kpi)};
            const projected = {json.dumps(quarterly_prs_per_eng_projection)};
            const lines = [`${{totals[i]}} KPI PRs / ${{counts[i]}} engineers`];
            if (projected[i] !== null && projected[i] !== undefined) {{
              lines.push(`Projected finish: ${{projected[i]}} PRs/engineer`);
            }}
            return lines;
          }}
        }}
      }}
    }}
  }},
}});

// Quarterly totals
new Chart(document.getElementById('qtrTotalsChart'), {{
  type: 'bar',
  data: {{
    labels: qtrLabels,
    datasets: [
      {{ label: 'Human PRs', data: {json.dumps(quarterly_total_merged)}, backgroundColor: '#4e79a7' }},
      {{ label: 'AI PRs', data: {json.dumps(quarterly_total_ai)}, backgroundColor: '#b07aa1' }},
    ],
  }},
  options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, x: {{ ...chartDefaults.scales.x, stacked: true }}, y: {{ ...chartDefaults.scales.y, stacked: true }} }} }},
}});
{autonomous_js}</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print()
    print(f"  📊 HTML report saved to: {output_path}")


if __name__ == "__main__":
    main()
