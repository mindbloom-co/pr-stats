# pr-stats

Mindbloom engineering KPI dashboard. `pr-metrics.py` builds `pr-metrics-report.html`
and copies it to `index.html`, which GitHub Pages serves at
https://mindbloom-co.github.io/pr-stats/.

## Usage

```sh
# Live from GitHub (needs an authenticated `gh`)
python pr-metrics.py 2026-01-01 2026-11-04 --history pr-metrics-history.json --collect-from 2026-07-01

# Offline, from pre-collected PRs (this is how the current report is built)
python pr-metrics.py 2026-01-01 2026-11-04 \
    --history pr-metrics-history.json --collect-from 2026-07-01 \
    --input data/q3-2026-human-merged.json \
    --input data/q4-2026-human-merged-through-2026-10-21.json \
    --input data/devin-q3-2026.json \
    --input data/devin-q4-2026-through-2026-10-21.json \
    --input data/q4-2026-all-merged-through-2026-11-04.json \
    --as-of "2026-11-04 ~9:02 AM CT"
```

- `--history`: frozen per-cycle aggregates (Q1-Q2) for earlier cycles.
- `--collect-from`: collect from this date even where history exists.
- `--as-of`: when the data was retrieved. Marks the cutoff day as partial.
- `--input` (repeatable): pre-collected merged PRs. Skips `gh`.

## Input format

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
  count in the denominator and show 0 labeled.
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
  An input must list **all** merged PRs, bots included, from the start of the quarter
  for the quarter-to-date share to be right. `data/q4-2026-all-merged-through-2026-11-04.json`
  has this for Q4 2026; the older `data/*human*` and `data/devin*` files do not.
