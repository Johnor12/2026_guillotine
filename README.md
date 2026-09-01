# 2026 redraft

A toolkit for a 32-team 0.5 PPR guillotine redraft league on Sleeper
("Gnosis Guillotine", league 1397662420398247936). Independent processes publish
stable JSON artifacts at the repository root; the ranker consumes those artifacts
and the static dashboard renders the result.

## League assumptions

- 0.5 PPR with a +1.0/rec tight end premium, 1 QB (superflex arrives in week 14)
- Guillotine: the two lowest weekly scores are eliminated each of weeks 1–15 and
  their players go to FAAB waivers; the last two teams play a week 16–17
  total-points championship
- Opening starters: 1 QB, 1 RB, 2 WR, 1 TE, 2 W/R/T flex — no D/ST or kicker slot;
  lineups expand in-season (+1 WR wk 7, +1 RB wk 9, +1 flex wk 12, +1 superflex
  wk 14, bench grows from 1 to 5)
- 2 reserve spots (reserve is not drafted into)
- No per-position roster caps
- 32 teams and 8 drafted players per team (256 picks, all offense)
- Snake draft with a third-round reversal: round 1 forward, rounds 2–3 reversed,
  alternating from there; picks can be traded
- My slot is 20 (johnor): 1.20, 2.13, 3.13, 4.20, 5.13, 6.20, 7.13, 8.20 before trades
- The guillotine is the objective: a roster is valued week by week (byes and known
  absences are zero weeks, lineups take each week's actual shape) and the weeks are
  combined by survival-hazard weights from a simulated elimination race — see the
  [ranker](ranker/README.md)

These are project assumptions, not runtime configuration. Ranker constants live in
`ranker/league.py`; the draft's actual geometry (teams, rounds, reversal, my slot) is
adopted from `draft.json` on every run.

## Setup

[uv](https://docs.astral.sh/uv/) pins Python 3.12.

```bash
uv sync
uv run <script>
```

Commands work from the repository root. Pipeline defaults are anchored to their own
directories, so their documented direct invocations also work from inside the pipeline.

## Data flow

```text
pool_pipeline/ ───────────────> pool.json ──────────┐
                                                   │
draft_pipeline/ ──────────────> draft.json ─────────┼─> rank.py ────> rankings.json
                                                   │
data_source_investigator/ ────> data_source_matches.json
                         └────> data/rankings.json ─┘
```

The published files have distinct owners:

- `pool.json`: ~370 QB/RB/WR/TE players keyed to Sleeper, priced by DraftSharks'
  league-scored per-week projections for weeks 1–17, which carry byes and known
  absences as zero weeks (identity and ADP from DraftSharks; Sleeper's season
  projection remains as the pool-membership filter)
- `draft.json`: all 256 made and pending picks from Sleeper's draft API, which is
  public and real-time — a fetch between picks is current
- `data_source_matches.json`: the provider board closest to each opponent's picks
- `rankings.json`: undrafted-player rankings, recommendations, simulations, and validation

`sleeper_id` is the cross-process player key. `roster_id` and `draft_slot` connect
opponent source matches to the live board.

## Components

- [Pool pipeline](pool_pipeline/README.md): provider HTML to the league-specific pool
- [Draft pipeline](draft_pipeline/README.md): Sleeper's draft API to the complete live board
- [Data-source investigator](data_source_investigator/README.md): normalize provider
  boards and infer opponent strategies
- [Ranker](ranker/README.md): wire-level solver, opponent simulation, planning,
  and output contracts
- `index.html`: dependency-free dashboard for `rankings.json`
- `data_source_investigator/index.html`: source-fit and pick-evidence dashboard
- `serve.py`: serves both dashboards at http://127.0.0.1:8123

Each component keeps its own paths, entry points, and implementation context. Offline
checks remain beside the draft, investigator, and ranker code they exercise. The pipelines
meet through their published JSON contracts rather than shared orchestration.

The ranker values a roster as guillotine-weighted expected weekly lineup points: each
week's expected optimal lineup under that week's starting shape and per-week
projections, weighted by the marginal effect of a weekly point on surviving that
week's cut (championship weeks by winning the final). Position-wide availability
determines when depth is called on, and per-week tiered waiver bodies — the best
undrafted player early, fresh eliminated-roster drops later — supply the fallback.
Personal and opponent strategies are intentionally separate: my slot uses the
projection-based roster objective, while each opponent follows its inferred external
board with roster-balance adjustments and fitted choice noise. Opponent picks never
use my projections or board.

## Common workflows

Rebuild the projection pool after saving updated provider HTML:

```bash
uv run pool_pipeline/pipeline.py --report
```

Re-price the pool after refetching Sleeper projections:

```bash
uv run pool_pipeline/fetch_sleeper_projections.py
uv run pool_pipeline/pipeline.py --report
```

Refresh ranking snapshots and opponent associations:

```bash
uv run data_source_investigator/pipeline.py --report
```

Refresh the live board and recommendations between picks — Sleeper's draft API is
real-time, so this is one command:

```bash
uv run refresh.py --report
```

`refresh.py` deliberately does only three live steps: fetch the draft, re-run source
inference against the existing provider snapshot, then rank. It does not rebuild the
offline pool or fetch provider boards.

To follow a different draft (e.g. a league mock at `sleeper.com/draft/nfl/<id>`),
fetch it explicitly first, then run the investigator and ranker:

```bash
uv run draft_pipeline/fetch_draft.py --draft-id <id>
uv run data_source_investigator/investigate.py
uv run rank.py --report
```

Run offline checks:

```bash
uv run rank.py --selftest
uv run draft_pipeline/fetch_draft.py --selftest
uv run data_source_investigator/investigate.py --selftest
uv run evaluate_opponents.py
```

Before and after changing an opponent model or pick policy, run
`uv run evaluate_opponents.py` and compare its replay accuracy.

## Dashboard

Run `uv run serve.py` and open the local URL; direct `file://` access cannot fetch the
JSON files. The dashboard renders the `rankings.json` snapshot — including the live
board state embedded in it — and shows when that snapshot was taken; re-run
`refresh.py` and reload to advance it.
