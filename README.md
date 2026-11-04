# pr-stats

Mindbloom engineering KPI dashboard. `pr-metrics.py` builds `pr-metrics-report.html`
and copies it to `index.html`, which GitHub Pages serves at
https://mindbloom-co.github.io/pr-stats/.

> mindbloomDbt, mindbloom-data and mindbloom-data-mcp included from Q4'26 (2026-10-01); earlier quarters unchanged.

## Usage

Refresh (live from GitHub, needs an authenticated `gh` that can read the repos in
`REPOS`). Set END to today's date in America/Chicago and `--as-of` to the retrieval time:

```sh
python pr-metrics.py 2026-01-01 END \
    --history pr-metrics-history.json --collect-from 2026-07-01 \
    --as-of "END ~H:MM AM CT"
# example: python pr-metrics.py 2026-01-01 2026-11-04 --history pr-metrics-history.json \
#              --collect-from 2026-07-01 --as-of "2026-11-04 ~9:30 AM CT"
```

The run takes a few minutes. It rewrites `pr-metrics-report.html` and `index.html`.
Commit both, with `pr-metrics.py` or the history file if they changed, and push to `main`.

- `--history`: frozen per-cycle aggregates (Q1-Q2) for earlier cycles.
- `--collect-from`: collect live from this date even where history exists.
- `--as-of`: when the data was retrieved. Marks the cutoff day as partial.
- Repos: `REPOS` in `pr-metrics.py`. `REPO_EFFECTIVE_START` gives a first date per repo
  (mindbloomDbt, mindbloom-data, mindbloom-data-mcp: 2026-10-01). Before that date the repo
  is not queried: merged PRs before it, and opened PRs created before it, are not counted.
  A cycle that crosses the date counts the repo only from the date. Q1-Q3 2026 do not change.
- `--input` (optional, repeatable): pre-collected merged PRs instead of live
  collection. Skips `gh`. Fallback only (see "Input format"); the report is built live.

## Input format (optional fallback)

A JSON list of merged PRs, or `{"merged": [...], "opened": [...]}`. Each PR has
`author`, `created`/`createdAt`/`created_at`, `merged`/`mergedAt`/`merged_at`, and
optionally:

| Field | Meaning |
| --- | --- |
| `repo`, `n`/`number` | Used to drop duplicates across input files. The first file that has a PR wins for every KPI field. |
| `reviewers` | Reviewer logins. |
| `is_ai_pr` | Devin PR flag. Defaults from the author. |
| `shepherd` / `credited_to` | Human credited for a Devin PR. |
| `all_approvals` | `[login, state, submitted_at]` per approval. Devin PRs are credited to the first human approver. Also used by the autonomous-merge cross-check. |
| `labels` | PR label names, as strings or `{"name": ...}`. Missing means no labels (older files stay valid). |

If a PR appears in several input files, its `labels` are merged and `all_approvals`
is filled in from a later file when the first file has none. Nothing else is changed.

## KPIs

These rules are fixed. The autonomous-merge metric below does not change them.

- **PRs/engineer**: human-authored merged PRs, plus merged Devin PRs credited to
  their first human approver, divided by engineers with at least one credited PR in
  the quarter. Dependabot and other bots are excluded. `gracesliu` is excluded.
  Devin PRs with no human approval are excluded and shown as a count.
- **Time to merge**: PR created to merged, human-authored PRs only. P90 is the
  headline (Q4 2026 goal: 72 hours). P50, P75 and P95 are context.

## Autonomous merges

Share of merged PRs that carry the `autonomous-merge` label (authored, reviewed and
queued by agents with no human review). Tracking starts 2026-10-21 (first labeled
merge: mindbloomBackend #6188). No goal is set for Q4 2026.

```
autonomous share = merged PRs labeled autonomous-merge
                   / all merged PRs in the configured repos in the same window
```

- Window dates are UTC merge dates, the same as the rest of the report.
- The numerator and denominator **include** Dependabot, other bot and Devin PRs.
  This is the only metric that does. `gracesliu` is excluded, as in the KPIs.
- The label is read from all configured repos (`REPOS` in the script). It exists on
  mindbloomBackend, mindbloomFrontend and mindbloomBackOffice today; the other repos
  count in the denominator and show 0 labeled. mindbloomDbt, mindbloom-data and
  mindbloom-data-mcp are in the denominator from 2026-10-01 (Q4'26).
- The report shows the share since the tracking start, the quarter to date, the share
  per reporting cycle (2-week cycles, the first one starting at the tracking date), a
  per-repo table and a per-author table (`linuxlewis`, `dependabot`, other).
  Quarter to date counts merges before the tracking start as unlabeled, because the
  label did not exist yet.
- **Cross-check** (human approval = an `APPROVED` review submitted before the merge by
  someone who is not a bot, Devin or the merge queue app):
  - (a) labeled PRs merged with no human approval.
  - (b) unlabeled PRs merged with no human approval. This can point to under-labeling,
    for example queued PRs.

  PRs whose approvals were not collected are left out of both counts and reported.
- Live collection reads `labels(first: 20)` and `reviews(first: 50)` for every PR.
  With `--input`, an input must list **all** merged PRs, bots included, from the start
  of the quarter for the quarter-to-date share to be right.
