# 2026 guillotine

A draft toolkit for a 32-team 0.5 PPR guillotine redraft league on Sleeper
("Gnosis Guillotine", league 1397662420398247936). Four processes each publish one JSON
artifact at the repository root; the ranker consumes the other three and the static
dashboard renders its output.

## League assumptions

- 0.5 PPR with a +1.0/rec tight end premium, 1 QB (superflex arrives in week 14)
- Guillotine: the two lowest weekly scores are eliminated each of weeks 1–15 and
  their players go to FAAB waivers; the last two teams play a week 16–17
  total-points championship
- Opening starters: 1 QB, 1 RB, 2 WR, 1 TE, 2 W/R/T flex; no D/ST or kicker slot;
  lineups expand in-season (+1 WR wk 7, +1 RB wk 9, +1 flex wk 12, +1 superflex
  wk 14, bench grows from 1 to 5)
- 2 reserve spots (reserve is not drafted into), no per-position roster caps
- 32 teams and 8 drafted players per team (256 picks, all offense)
- Snake draft with a third-round reversal: round 1 forward, rounds 2–3 reversed,
  alternating from there; picks can be traded
- My slot is 20 (johnor): 1.20, 2.13, 3.13, 4.20, 5.13, 6.20, 7.13, 8.20 before trades

These are constants in `ranker/league.py`, not runtime configuration. The ranker
complains loudly when `draft.json` disagrees with them.

## Setup

[uv](https://docs.astral.sh/uv/) pins Python 3.12; everything is stdlib.

```bash
uv sync
uv run <script>
```

Every script anchors its paths to its own location, so commands work from anywhere.

## Data flow

```text
pool/    projections.html + sleeper_projections.json + weekly_projections.json
         └─ build_pool.py ─────────────────────────────> pool.json ──────────────┐
sources/ fetch_rankings.py -> data/raw/ -> build_rankings.py -> data/boards.json ─┤
                                                                                  │
draft/   fetch_draft.py ──────────────────────────────> draft.json ──────────────┤
                                                                                  │
sources/ investigate.py (boards + draft) ─────────────> data_source_matches.json ─┤
                                                                                  │
rank.py ──────────────────────────────────────────────> rankings.json <───────────┘
```

- `pool.json`: ~370 QB/RB/WR/TE players. DraftSharks supplies identity, age, bye,
  rookie flag and 1QB ADP from a hand-saved rankings page; a Sleeper season projection
  gates membership and supplies `sleeper_id`; `weekly_points` is DraftSharks' per-week
  projection for weeks 1–17 in the league's scoring, with byes and known absences as
  zero weeks. It is the ranker's value input, after `ranker/market.py` blends each
  season total equally with Sleeper's projection (`points`) and the consensus board's
  rank-matched level within the position.
- `draft.json`: all 256 made and pending picks from Sleeper's public, real-time draft
  API, with pending picks derived from the draft settings and traded picks applied.
- `sources/data/boards.json`: provider boards (FantasyCalc, KeepTradeCut, FF Calculator
  ADP, FantasyPros ECR, DraftSharks ADP, Sleeper ADP, and a consensus average),
  each row resolved to the pool's `sleeper_id`.
- `data_source_matches.json`: for each drafter, the board closest to its picks so far,
  with fit scores and pick-level evidence.
- `rankings.json`: undrafted-player rankings, next-pick recommendations, the example
  draft, and validation.

`sleeper_id` is the cross-process player key; `roster_id` and `draft_slot` connect
opponent source matches to the live board.

## Method

The guillotine is the objective. A roster is valued week by week as the expected
optimal lineup under that week's starting shape and position-wide availability, using
per-week projections, and the 17 weekly values are combined by guillotine week weights:
each week's weight is the marginal effect of a weekly point on log P(surviving that
week's cut), measured by simulating the elimination race over the opponents' simulated
rosters, with the championship weeks entering through log P(winning the final). The
waiver wire is per week and tiered: the undrafted tail early, then the survivors' equal
split of every roster eliminated so far, so by the final the wire is other teams'
first-round picks and drafted depth is worth nothing while drafted stars still clear
it. Levels and the simulated draft are a fixed point that converges to a limit cycle.
My slot alone uses this objective, on projections blended toward the market so the
draft does not build around one source's outliers; each opponent follows the external
board most associated with its picks, with roster-balance adjustments, fitted choice
noise, and prior tilts for what no 1QB no-TE-premium board prices (QB scarcity across
32 teams, the TE premium), and never sees my projections. The first pending decision
searches target plans across my next four held picks and plays each plan out to the
end of the draft. See `rank.py` and the `ranker/` module docstrings for the details.

## Workflows

Refresh the live board and recommendations between picks (Sleeper's draft API is
real-time, so this is the whole live loop):

```bash
uv run refresh.py --report
```

It runs three steps: fetch the draft, re-run source inference against the existing
boards snapshot, then rank. It never rebuilds the pool or fetches provider boards.

Rebuild the pool after refetching projections or saving a new DraftSharks page to
`pool/data/projections.html`:

```bash
uv run pool/fetch_projections.py      # Sleeper season + DraftSharks weekly, manual
uv run pool/build_pool.py --report
```

Refresh the provider boards (rebuild the pool first if it changed, since every board
resolves onto it):

```bash
uv run sources/fetch_rankings.py
uv run sources/build_rankings.py --report
```

Follow a different draft, such as a league mock at `sleeper.com/draft/nfl/<id>`:

```bash
uv run draft/fetch_draft.py --draft-id <id>
uv run sources/investigate.py
uv run rank.py --report
```

Offline checks:

```bash
uv run rank.py --selftest
uv run draft/fetch_draft.py --selftest
uv run sources/investigate.py --selftest
uv run evaluate_opponents.py   # replays every completed opponent pick through the model
```

Before and after changing the opponent model, compare `evaluate_opponents.py`'s replay
accuracy.

## Dashboards

`uv run serve.py` serves the repository at http://127.0.0.1:8123 (direct `file://`
access cannot fetch the JSON). `/` renders `rankings.json`, including the live board
state embedded in it and when that snapshot was taken; `/sources/` renders
`data_source_matches.json` as a team-by-source fit heatmap with pick-level evidence.
Re-run `refresh.py` and reload to advance either.
