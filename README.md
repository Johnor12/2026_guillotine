# 2026 guillotine

A draft and in-season toolkit for a 32-team 0.5 PPR guillotine redraft league on
Sleeper ("Gnosis Guillotine", league 1397662420398247936). Four draft processes each
publish one JSON artifact at the repository root; the ranker consumes the other three
and the static dashboard renders its output. Two more processes run the season: a
league-state fetch and the season model behind the season desk.

## League assumptions

- 0.5 PPR with a +1.0/rec tight end premium, 1 QB (superflex arrives in week 14)
- Guillotine: the two lowest weekly scores are eliminated each of weeks 1–15 and
  their players go to FAAB waivers; the last two teams play a week 16–17
  total-points championship
- Opening starters: 1 QB, 1 RB, 2 WR, 1 TE, 2 W/R/T flex; no D/ST or kicker slot;
  lineups expand in-season (+1 WR wk 7, +1 RB wk 9, +1 flex wk 12, +1 superflex
  wk 14, bench grows from 1 to 5, assumed one spot at each expansion)
- $1000 FAAB, claims processed once a week, unclaimed players free afterwards
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

season/  fetch_league.py (Sleeper rosters, FAAB, moves, weekly projections) -> league.json ─┐
pool/data/weekly_projections.json + pool.json (id join) ────────────────────────────────────┤
season.py ────────────────────────────────────────────> season.json <───────────────────────┘
```

- `pool.json`: ~370 QB/RB/WR/TE players. DraftSharks supplies identity, age, bye,
  rookie flag and 1QB ADP from a hand-saved rankings page; a Sleeper season projection
  gates membership and supplies `sleeper_id`; `weekly_points` is DraftSharks' per-week
  projection for weeks 1–17 in the league's scoring, with byes and known absences as
  zero weeks. It is the ranker's value input, after `ranker/projections.py` blends
  each season total 2:1 with Sleeper's league-scored projection (`points`).
- `draft.json`: all 256 made and pending picks from Sleeper's public, real-time draft
  API, with pending picks derived from the draft settings and traded picks applied.
- `sources/data/boards.json`: provider boards in two families, ordinary 1QB redraft
  (FantasyCalc, KeepTradeCut, FF Calculator ADP, FantasyPros ECR, DraftSharks ADP,
  Sleeper ADP) and the superflex/2QB variant of each provider for this room's QB
  scarcity, plus Sleeper's league-scored points order and a value-over-replacement
  board on that projection under the opening lineup across 32 teams (the two boards
  that price the TE premium; the VORP board is what an LLM handed the league id and
  Sleeper's API arrives at), a consensus average, and the `cold_start` room prior:
  50% Sleeper's half-PPR ADP, which is what the draft room displays and autopick
  drafts from, 30% the VORP board for the LLM-assisted minority this office is
  expected to hold, and 20% the format-adjusted boards. Each row is resolved to the
  pool's `sleeper_id`.
- `data_source_matches.json`: for each drafter, the board closest to its picks so far,
  with fit scores and pick-level evidence.
- `rankings.json`: undrafted-player rankings, next-pick recommendations, the example
  draft, and validation.
- `league.json`: the in-season state from Sleeper: the current NFL week, every roster
  with its players, set starters, reserve, FAAB used and points, every waiver claim and
  free-agent move so far with its bid, a directory of every referenced player (name,
  position, team, injury status), and Sleeper's weekly projections in league scoring for
  the current week through week 17. A roster the commissioner has emptied after week 1
  is an eliminated team (Sleeper has no guillotine flag).
- `season.json`: this week's optimal lineup and the moves it implies, the optimal FAAB
  bid on each free agent worth a look, every team's chance of being cut this week, of
  reaching the final and of the title, its expected spend and budget path, the
  elimination bar by week, and the observed versus simulated waiver market.

`sleeper_id` is the cross-process player key; `roster_id` and `draft_slot` connect
opponent source matches to the live board.

## Method

The guillotine is the objective. A roster is valued week by week as the expected
optimal lineup under that week's starting shape and position-wide availability, using
per-week projections, and the 17 weekly values are combined by guillotine week weights:
each week's weight is the marginal effect of a weekly point on log P(surviving that
week's cut), measured by simulating the elimination race over the opponents' simulated
rosters, with the championship weeks entering through log P(winning the final). The
waiver wire is per week and tiered: one free-agent pool, the undrafted tail plus every
roster eliminated so far, split equally among the survivors, so in week 1 a lone
undrafted starter is worth a thirty-second of himself and by the final the wire is
other teams' first-round picks; drafted depth is worth nothing by then while drafted
stars still clear it. That equal split prices the opponents and so the elimination
bars; my own roster is priced under my FAAB policy, since the draft should not assume
free claims I do not intend to make: the undrafted tail alone through week 8 while I
hold the budget, the equal split through the week 9-12 lineup expansion, and the top
half of every tier from week 13 once the saved budget is spent. Levels and the
simulated draft are a fixed point that converges to a limit cycle.
My slot alone uses this objective, on DraftSharks projections blended 2:1 with
Sleeper's so the draft does not build around one model's outliers; each opponent
follows the external board most associated with its picks (the `cold_start` blend
until it has any), with roster-balance adjustments, fitted choice noise, a residual
TE tilt for the share of the room that half-notices the premium without a board
that prices it, and QB scarcity sense (a team without a quarterback never takes one
who does not start week 1, and takes a starter once the run leaves none likely to
last to its next pick), and never sees my projections. A drafter named in
`OPPONENT_INTEL` (`ranker/league.py`) follows its stated plan for its next picks
before any of that applies. The first pending decision
searches target plans across my next four held picks and plays each plan out to the
end of the draft. See `rank.py` and the `ranker/` module docstrings for the details.

## In-season method

The season desk runs an agent-based race from the live state (`ranker/race.py`). The
value input is per week: DraftSharks' weekly projection blended 2:1 with Sleeper's for
the same week, Sleeper alone for anyone outside the draft pool (`ranker/season.py`).
Each simulated week every alive team fields its greedy optimal lineup (exact for this
slot chain) under the draft model's noise, the two lowest are cut and their players hit
the wire, and before the next games the survivors bid FAAB: each team prices the top
free agents by their lineup gain over the starter they would displace and claims its
best few at a share of its remaining budget, convex in that gain, tempered early in the
season, with noise; claims resolve highest bid first and whatever clears unclaimed is a
free pickup (`ranker/league.py`, `CLAIM_*`). Week 1 is free agency. Run without me the
race records the elimination bars and the market I face, and my roster variants are
replayed through those records in closed form (`ranker/claims.py`): for each free agent,
the bid that maximizes P(win at that price) x title odds with him at that budget +
P(lose) x title odds standing pat, with the room's clearing prices from the same
seasons. Both of my FAAB policies (hold the budget until week 9, or bid like the room)
are replayed and the better one governs my future claims. Run with all 32 the race is
every team's cut, final and title odds. The claim constants are a prior; the desk shows
the room's observed bids beside the simulated ones so they can be refit as weeks pass.

## Workflows

Refresh the season desk (Tuesday or Wednesday before claims process for the week's
bids; after lineups lock for the week's cut odds). No manual input:

```bash
uv run refresh_season.py --report
```

It refetches projections (DraftSharks weekly for the weeks still to play, Sleeper
season), fetches the league state, and runs the season model, about a minute in all.

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
state embedded in it and when that snapshot was taken; `/season.html` renders
`season.json`: this week's lineup, the waiver claims with their optimal bids, the
elimination bar by week, every team's odds and budget outlook, and the waiver market;
`/sources/` renders `data_source_matches.json` as a team-by-source fit heatmap with
pick-level evidence. Re-run `refresh_season.py` or `refresh.py` and reload to advance.
