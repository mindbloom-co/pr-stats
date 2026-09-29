# Review Guidelines

This repository calculates Mindbloom engineering PR metrics. `pr-metrics.py` queries GitHub through the `gh` CLI, computes per-person and team statistics by two-week cycle and quarter, and writes an HTML report. `index.html` and `pr-metrics-report.html` are the published report. Review changes for metric accuracy, because the results are reported as a team KPI.

## How to review

- Judge the actual behavior of the change. Read the surrounding code and callers before filing a finding, and confirm the affected path is reachable.
- Cite a concrete convention from this repository (a doc, config, or nearby code) when a finding depends on local practice. Do not invent conventions; otherwise apply general best practice and say so.
- Explain each finding with a concrete failure scenario and a suggested direction. Prioritize issues that would break behavior, leak data, or mislead readers over stylistic preferences.
- Skip low-value formatting nits, especially anything a formatter, linter, or CI check already enforces.
- Do not ask for broad rewrites unless the change introduces a real risk.
- Call out verification gaps: behavior that is not covered by tests or checks, and risks that can only be confirmed outside the diff.
- Never repeat secrets, tokens, or personal or health data in review comments.

## What to check

- **Correctness:** logic matches the intended behavior; edge cases such as empty or missing data, partial failures, and unexpected states are handled; existing behavior is preserved unless the change intends otherwise; dates, time zones, currency, and units are explicit where relevant.
- **Error handling:** failures surface at the right boundary; errors are not swallowed; retries, fallbacks, and defaults do not hide corruption or user-visible failures.
- **Security, privacy, and authorization:** sensitive operations are authorized; inputs are validated before database, shell, template, or API use; secrets are not logged, committed, or returned; trust boundaries do not expand unexpectedly.
- **Performance:** no unbounded loops, unpaginated requests, N+1 patterns, or repeated expensive work; queries use appropriate filters and limits.
- **Concurrency and state:** duplicate actions, retries, races, and partial writes are considered.
- **Compatibility and contracts:** API, CLI, schema, config, and file-format changes stay backward-compatible or update every caller and consumer.
- **Tests:** tests exercise the changed behavior and important edge cases; mocks or snapshots do not hide regressions; missing coverage is called out when the risk justifies it.
- **Maintainability:** the change follows existing names, layering, and patterns; abstractions remove real duplication; dead code and stale comments or docs are removed.
- **Dependencies:** new dependencies and version or lockfile changes are intentional, maintained, and appropriate for the repository.

## Privacy and health data

This repository should only handle GitHub activity metadata. Flag changes that add tokens, private repository content, or any personal or health data from Mindbloom systems to the script output or the committed reports.

## Repository-specific checks

These come from `AGENTS.md`, `pr-metrics.py`, and the repository history.

- **Metric definitions:** check changes to cycle boundaries, date ranges, time zones, merged versus opened counts, review counts, and time-to-merge against the definitions in the script docstring. An off-by-one date boundary or double-counted PR changes the reported KPI.
- **Attribution and exclusions:** bot accounts in `BOTS` and users in `EXCLUDED_USERS` must stay excluded from all statistics. PRs authored by accounts in `AI_AGENTS` are attributed to the human who approved them. Check that new code paths apply the same rules. Grace stays excluded unless the instructions in `AGENTS.md` change.
- **GitHub API use:** GraphQL queries should paginate completely, handle errors and partial responses without silently producing low counts, and keep the existing retry behavior.
- **Report outputs:** `index.html` and `pr-metrics-report.html` are identical generated copies. Report refreshes should update both together from the script output, not by hand-editing one file. Check that the date range shown in the report matches the data it was generated from.
- **Configuration:** changes to `REPOS`, `CYCLE_DAYS`, or `QUARTERLY_PRS_PER_ENGINEER_TARGET` change the KPI and should be intentional and explained.
- **HTML generation:** values interpolated into the generated HTML should come from trusted sources or be escaped.
