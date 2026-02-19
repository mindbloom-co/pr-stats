#!/usr/bin/env python3
"""
Mindbloom PR Metrics Calculator

Calculates per-person PR authoring & reviewing stats across repos,
broken into 2-week cycles with cycle-over-cycle comparison.

Metrics:
  - PRs authored (merged) per person per cycle
  - PRs opened per person per cycle
  - Reviews given per person per cycle
  - Time to merge (created → merged) per person and team average

Usage:
    python pr-metrics.py                        # last 1 cycle (2 weeks)
    python pr-metrics.py 4                      # last 4 cycles (8 weeks)
    python pr-metrics.py 2025-01-01 2025-06-01  # custom date range

Requires: gh (GitHub CLI) authenticated with org access
"""

import json
import subprocess
import sys
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
]
CYCLE_DAYS = 14

# Bot accounts to exclude from all stats
BOTS = {
    "gemini-code-assist",
    "devin-ai-integration",
    "github-actions",
    "dependabot",
    "claude",
    "coderabbitai",
}


# ── GitHub API ──────────────────────────────────────────────

def gh_graphql(query: str) -> dict:
    """Run a GraphQL query via the gh CLI and return parsed JSON."""
    result = subprocess.run(
        ["gh", "api", "graphql", "-f", f"query={query}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def _paginated_search(search_query: str, extra_fields: str = "") -> list[dict]:
    """Generic paginated GitHub search returning PR nodes."""
    results = []
    has_next = True
    cursor = None

    while has_next:
        after = f', after: "{cursor}"' if cursor else ""
        query = f"""{{
            search(query: "{search_query}", type: ISSUE, first: 100{after}) {{
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
        if not data:
            break

        search_data = data.get("data", {}).get("search", {})
        page_info = search_data.get("pageInfo", {})
        results.extend(search_data.get("nodes", []))

        has_next = page_info.get("hasNextPage", False)
        cursor = page_info.get("endCursor")

    return results


def fetch_merged_prs(repo: str, start: str, end: str) -> list[dict]:
    """
    Fetch all merged PRs for a repo in a date range.
    Returns list of {author, reviewers, created_at, merged_at, time_to_merge_hours}
    """
    search_query = f"repo:{ORG}/{repo} is:pr is:merged merged:{start}..{end}"
    review_fields = """
        reviews(first: 50) {
            nodes {
                author { login }
                state
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
        for review in (node.get("reviews") or {}).get("nodes", []):
            reviewer = (review.get("author") or {}).get("login")
            if reviewer and review.get("state") in review_states:
                reviewers.add(reviewer)

        # Time to merge
        created_at = node.get("createdAt")
        merged_at = node.get("mergedAt")
        ttm_hours = None
        if created_at and merged_at:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            merged = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
            ttm_hours = (merged - created).total_seconds() / 3600

        prs.append({
            "author": author,
            "reviewers": list(reviewers),
            "created_at": created_at,
            "merged_at": merged_at,
            "ttm_hours": ttm_hours,
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

def is_human(username: str) -> bool:
    return username not in BOTS


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
    while current < end:
        cycle_end = min(current + timedelta(days=CYCLE_DAYS - 1), end)
        cycles.append((current, cycle_end))
        current += timedelta(days=CYCLE_DAYS)
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


# ── Main ────────────────────────────────────────────────────

def main():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Parse arguments
    args = sys.argv[1:]
    if len(args) == 1 and args[0].isdigit():
        num_cycles = int(args[0])
        end_date = today
        start_date = today - timedelta(days=num_cycles * CYCLE_DAYS)
    elif len(args) == 2:
        start_date = datetime.strptime(args[0], "%Y-%m-%d")
        end_date = datetime.strptime(args[1], "%Y-%m-%d")
    else:
        end_date = today
        start_date = today - timedelta(days=CYCLE_DAYS)

    cycles = build_cycles(start_date, end_date)

    print("═" * 60)
    print(f"  Mindbloom PR Metrics")
    print(f"  Period: {fmt_date(start_date)} → {fmt_date(end_date)} ({len(cycles)} cycles)")
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
        for repo in REPOS:
            s, e = fmt_date(cycle_start), fmt_date(cycle_end)
            cycle_merged.extend(fetch_merged_prs(repo, s, e))
            cycle_opened.extend(fetch_opened_prs(repo, s, e))

        all_cycle_merged.append(cycle_merged)
        all_cycle_opened.append(cycle_opened)

        # Filter to humans
        human_merged = [pr for pr in cycle_merged if is_human(pr["author"])]
        human_opened = [pr for pr in cycle_opened if is_human(pr["author"])]
        human_reviewers = [
            r for pr in cycle_merged for r in pr["reviewers"] if is_human(r)
        ]

        # Opened vs Merged
        opened_count = len(human_opened)
        merged_count = len(human_merged)
        print()
        print(f"  Opened: {opened_count}  |  Merged: {merged_count}")

        if not human_merged:
            print("  No merged PRs this cycle.")
            continue

        # PRs Authored (merged)
        print()
        print("  PRs Authored (merged):")
        for author, count in count_sorted([pr["author"] for pr in human_merged]):
            # Per-person TTM
            author_prs = [pr for pr in human_merged if pr["author"] == author]
            ttm = avg_ttm(author_prs)
            ttm_str = f"  avg merge: {fmt_duration(ttm)}" if ttm is not None else ""
            print(f"    {author:<25} {count:>3} PRs{ttm_str}")

        # PRs Opened
        print()
        print("  PRs Opened:")
        for author, count in count_sorted([pr["author"] for pr in human_opened]):
            print(f"    {author:<25} {count:>3} PRs")

        # Reviews
        print()
        print("  Reviews Given:")
        for reviewer, count in count_sorted(human_reviewers):
            print(f"    {reviewer:<25} {count:>3} reviews")

        # Cycle time-to-merge
        ttm_avg = avg_ttm(human_merged)
        ttm_med = median_ttm(human_merged)
        unique_authors = len(set(pr["author"] for pr in human_merged))
        print()
        print(f"  Cycle total: {merged_count} merged, {opened_count} opened, {unique_authors} contributors")
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
    human_opened = [pr for pr in all_opened if is_human(pr["author"])]
    human_reviewers = [r for pr in all_merged for r in pr["reviewers"] if is_human(r)]

    if not human_merged:
        print("  No PRs found.")
        return

    print()
    print(f"  Total opened: {len(human_opened)}  |  Total merged: {len(human_merged)}")

    print()
    print("  PRs Authored — merged (total | avg/cycle | avg merge time):")
    for author, count in count_sorted([pr["author"] for pr in human_merged]):
        avg = count / num_cycles
        author_prs = [pr for pr in human_merged if pr["author"] == author]
        ttm = avg_ttm(author_prs)
        ttm_str = fmt_duration(ttm) if ttm is not None else "n/a"
        print(f"    {author:<25} {count:>3} total   {avg:>5.1f}/cycle   merge: {ttm_str}")

    print()
    print("  PRs Opened (total | avg/cycle):")
    for author, count in count_sorted([pr["author"] for pr in human_opened]):
        avg = count / num_cycles
        print(f"    {author:<25} {count:>3} total   {avg:>5.1f}/cycle")

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
            print(f"  {fmt_date(cycle_start)} → {fmt_date(cycle_end)}  {len(o):>7} {len(m):>7} {ca:>8} {t_avg_s:>8} {t_med_s:>8}")

    # ── HTML Report ─────────────────────────────────────────
    import os
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr-metrics-report.html")
    generate_html_report(cycles, all_cycle_merged, all_cycle_opened, start_date, end_date, report_path)


def generate_html_report(
    cycles: list[tuple[datetime, datetime]],
    all_cycle_merged: list[list[dict]],
    all_cycle_opened: list[list[dict]],
    start_date: datetime,
    end_date: datetime,
    output_path: str = "pr-metrics-report.html",
):
    """Generate an interactive HTML report with Chart.js visualizations."""
    num_cycles = len(cycles)
    cycle_labels = [f"{fmt_date(cs)}→{fmt_date(ce)}" for cs, ce in cycles]

    # Collect all human authors across all cycles
    all_authors = sorted(set(
        pr["author"]
        for cycle in all_cycle_merged
        for pr in cycle
        if is_human(pr["author"])
    ))

    # ── Build data structures for charts ────────────────────

    # Per-person merged PRs per cycle
    merged_by_author_cycle = {a: [] for a in all_authors}
    opened_by_author_cycle = {a: [] for a in all_authors}
    reviews_by_author_cycle = {a: [] for a in all_authors}
    ttm_by_author_cycle = {a: [] for a in all_authors}

    trend_opened = []
    trend_merged = []
    trend_ttm_avg = []
    trend_ttm_med = []

    for i in range(num_cycles):
        human_merged = [pr for pr in all_cycle_merged[i] if is_human(pr["author"])]
        human_opened = [pr for pr in all_cycle_opened[i] if is_human(pr["author"])]

        trend_opened.append(len(human_opened))
        trend_merged.append(len(human_merged))
        t_avg = avg_ttm(human_merged)
        t_med = median_ttm(human_merged)
        trend_ttm_avg.append(round(t_avg, 1) if t_avg else 0)
        trend_ttm_med.append(round(t_med, 1) if t_med else 0)

        merged_counts = defaultdict(int)
        opened_counts = defaultdict(int)
        review_counts = defaultdict(int)
        ttm_sums = defaultdict(list)

        for pr in human_merged:
            merged_counts[pr["author"]] += 1
            if pr.get("ttm_hours") is not None:
                ttm_sums[pr["author"]].append(pr["ttm_hours"])
        for pr in human_opened:
            opened_counts[pr["author"]] += 1
        for pr in all_cycle_merged[i]:
            for r in pr["reviewers"]:
                if is_human(r):
                    review_counts[r] += 1

        for a in all_authors:
            merged_by_author_cycle[a].append(merged_counts.get(a, 0))
            opened_by_author_cycle[a].append(opened_counts.get(a, 0))
            reviews_by_author_cycle[a].append(review_counts.get(a, 0))
            ttms = ttm_sums.get(a, [])
            ttm_by_author_cycle[a].append(round(sum(ttms) / len(ttms), 1) if ttms else None)

    # Overall totals for summary cards
    all_human_merged = [pr for cycle in all_cycle_merged for pr in cycle if is_human(pr["author"])]
    all_human_opened = [pr for cycle in all_cycle_opened for pr in cycle if is_human(pr["author"])]
    overall_ttm_avg = avg_ttm(all_human_merged)
    overall_ttm_med = median_ttm(all_human_merged)

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
</style>
</head>
<body>

<h1>📊 Mindbloom PR Metrics</h1>
<p class="subtitle">{fmt_date(start_date)} → {fmt_date(end_date)} · {num_cycles} cycle{"s" if num_cycles != 1 else ""} · {len(all_authors)} contributors</p>

<div class="cards">
  <div class="card"><div class="value">{len(all_human_opened)}</div><div class="label">PRs Opened</div></div>
  <div class="card"><div class="value">{len(all_human_merged)}</div><div class="label">PRs Merged</div></div>
  <div class="card"><div class="value">{len(all_authors)}</div><div class="label">Contributors</div></div>
  <div class="card"><div class="value">{fmt_duration(overall_ttm_avg) if overall_ttm_avg else 'n/a'}</div><div class="label">Avg Time to Merge</div></div>
  <div class="card"><div class="value">{fmt_duration(overall_ttm_med) if overall_ttm_med else 'n/a'}</div><div class="label">Median Time to Merge</div></div>
  <div class="card"><div class="value">{len(all_human_merged) / len(all_authors) / num_cycles:.1f}</div><div class="label">Avg PRs/Person/Cycle</div></div>
</div>

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
    <h2>Reviews Given by Author (per cycle)</h2>
    <canvas id="reviewsChart"></canvas>
  </div>
  <div class="chart-box">
    <h2>Avg Time to Merge by Author (hours)</h2>
    <canvas id="ttmAuthorChart"></canvas>
  </div>
</div>

<div class="chart-box" style="margin-bottom: 32px;">
  <h2>Overall Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Author</th>
        <th class="num">Opened</th>
        <th class="num">Merged</th>
        <th class="num">Reviews</th>
        <th class="num">Avg Merge Time</th>
        <th class="num">Merged/Cycle</th>
      </tr>
    </thead>
    <tbody>
"""

    # Build table rows
    overall_merged_counts = defaultdict(int)
    overall_opened_counts = defaultdict(int)
    overall_review_counts = defaultdict(int)
    overall_ttm_by_author = defaultdict(list)

    for pr in all_human_merged:
        overall_merged_counts[pr["author"]] += 1
        if pr.get("ttm_hours") is not None:
            overall_ttm_by_author[pr["author"]].append(pr["ttm_hours"])
    for pr in all_human_opened:
        overall_opened_counts[pr["author"]] += 1
    for cycle in all_cycle_merged:
        for pr in cycle:
            for r in pr["reviewers"]:
                if is_human(r):
                    overall_review_counts[r] += 1

    for author in sorted(all_authors, key=lambda a: -overall_merged_counts.get(a, 0)):
        mc = overall_merged_counts.get(author, 0)
        oc = overall_opened_counts.get(author, 0)
        rc = overall_review_counts.get(author, 0)
        ttms = overall_ttm_by_author.get(author, [])
        ttm_str = fmt_duration(sum(ttms) / len(ttms)) if ttms else "n/a"
        avg_per_cycle = mc / num_cycles
        html += f"""      <tr>
        <td>{author}</td>
        <td class="num">{oc}</td>
        <td class="num">{mc}</td>
        <td class="num">{rc}</td>
        <td class="num">{ttm_str}</td>
        <td class="num">{avg_per_cycle:.1f}</td>
      </tr>\n"""

    html += f"""    </tbody>
  </table>
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
      {{ label: 'Merged', data: {json.dumps(trend_merged)}, borderColor: '#4e79a7', backgroundColor: 'rgba(78,121,167,0.1)', fill: true, tension: 0.3 }},
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
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    print()
    print(f"  📊 HTML report saved to: {output_path}")


if __name__ == "__main__":
    main()
