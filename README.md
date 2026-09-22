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
- 2 reserve spots holding Out/IR/PUP players (reserve is not drafted into), no
  per-position roster caps
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

## CPU limits

Draft and season simulation pools use at most **six worker processes**, limited
further by CPU affinity when available (`ranker/workers.py`). Simulation counts,
candidate searches, and seeded results are unchanged; runs take longer instead of
using every CPU. This is a limit per process pool, so run one pipeline at a time.

On this Windows/WSL desktop, `C:\Users\johnm\.wslconfig` also sets a limit for the
whole WSL 2 VM, shared across its distributions:

```ini
[wsl2]
processors=6
```

That file lives outside the repository. The limit takes effect after WSL shuts down
and restarts. Save work in WSL, run `wsl --shutdown` from **Windows PowerShell**, then
reopen the terminal; `nproc` should report `6`. Even programs that request more
workers must share those six virtual CPUs. This does not limit native Windows
programs. See [Microsoft's WSL configuration reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

These limits reduce load; they do not repair a machine that abruptly powers off
under load. The September 15 shutdown logged Kernel-Power event 41 with no bugcheck
code, which does not establish the cause. Check CPU cooling, power supply, and any
overclock/undervolt settings before trying sustained full-load tests; see
[AMD's stability troubleshooting](https://www.amd.com/en/resources/support-articles/faqs/PIBRMATS2.html).

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
pool/data/weekly_projections.json + pool.json (id join; name fallback via league.json) ────┤
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
  Transactions retain Sleeper's submission `leg`, ID and processing timestamp;
  `week` follows processing time relative to Sleeper's season-start date. Wednesday
  claims may remain under the previous submission leg. Both successful and failed
  bids are retained; pending claims do not establish that waivers have processed.
- `season.json`: this week's optimal lineup and the moves it implies, the optimal FAAB
  bid on each free agent worth a look with the drop and any move onto reserve it needs,
  every team's chance of being cut this week, of
  reaching the final and of the title, its expected spend and budget path, the
  elimination bar by week, and the observed versus simulated waiver market.
  Budget paths show cash entering each week, before that week's claims, conditional
  on surviving to that week. The current week's opening cash adds back completed
  claims to the live balance. Post-claim cash is recorded separately. Already
  eliminated teams have zero survival odds and no future budget estimates, and are
  excluded from league averages.

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
the same week, Sleeper alone for anyone DraftSharks does not project
(`ranker/season.py`). DraftSharks rows join Sleeper ids through `pool.json`; a row
outside the draft pool, such as a backup who became a starter, joins `league.json`'s
directory by normalized name and position when that names exactly one player.
Each simulated week every alive team fields its optimal lineup under the draft model's
noise, the two lowest are cut, and their players hit the wire. `ranker/waivers.py`
prices acquisitions from [Paul Charchian's guillotine FAAB guide](https://www.fantasylife.com/articles/guillotine-leagues/guillotine-league-fantasy-football-waiver-wire-guide-for-week-2):
early elite players about 15–20% of the starting budget, ordinary starters 2.5–5%,
and depth 0.1–1%. This is an 18-team guide, not a fitted rule for our 32-team format.
Our adaptation uses league-scored positional ranks by points per game played for the
rest of the season (elite anchors QB4/RB6/WR6/TE3, then an inverse-square price
curve), the claim's projected net lineup points through Week 17, per remaining week,
and projected cut risk. Ranking by games played keeps a player returning from an
absence from being charged twice: the net lineup points already count only the weeks
he plays. Each candidate's drop is chosen with him: the body whose loss leaves the best
remaining-season roster with the candidate on it, including distant byes and
superflex, so a backup QB goes when a better QB arrives rather than a bench RB. The
drop's whole remaining season is charged, so a returning starter is not a free
placeholder for a short-term fill-in; paired replays favor this over charging only
through the fill-in's best hold horizon. The two reserve slots hold Out/IR/PUP
bodies while their projection is zero; once a body's projection resumes he needs a
regular spot, and every simulated team cuts its least valuable body to make
room before that week's claims. The guide ceiling scales with remaining
cash and weeks, but a separate saving plan limits total auction spending.
Terminal-week improvements can use all remaining money.

Opponents learn separate participation probabilities and bid multipliers from their
submitted bids, including losses. Duplicate team/player/week claims use the latest
submission; a winner's roster and budget are rolled back before measuring his need.
Bid multipliers are shrunk toward the room and the published prior, with persistent
manager uncertainty and bid noise. A manager with no bids remains uncertain, and his
participation probability rises toward the end of the season. This reactivation curve
is an assumption, not something one auction can estimate. Earlier bids use current
projections as a proxy for historical player value. Manager estimates and held-out
bid-size errors are published in `season.json` under `market`.

Opponent saving habits are a uniform prior over three persistent season-long plans:
no reserve, balanced saving, and patient saving. The balanced plan targets 75% of
cash entering Week 9, 25% entering Week 13, and 20% entering Week 14, informed by
[Charchian's month-by-month guidance](https://www.fantasylife.com/articles/guillotine-leagues/how-to-manage-your-faab-in-guillotine-league-fantasy-football).
The patient plan targets 85%, 55%, and 50%, respectively, to preserve buying power
for late chopped rosters and superflex. Targets interpolate between milestones and
rescale to the manager's live remaining cash; unused allowances carry forward.
Reserves relax as projected cut risk rises from 25% to 50%, and reach zero after
Week 17. Bid multipliers and noise cannot exceed the resulting auction allowance.
These habits and their equal prior weights are assumptions, not inferred from one
auction or optimized for opponents. Participation, target noise and observed bid
tendencies still distinguish managers.

Our future policy submits offers for every improving candidate within a team-specific
ceiling and saving allowance, including early bargains. Total paid spending in an
auction is also limited to its largest individual bid. Claims naming the same drop
are alternatives; open spots, remaining cash and redundant upgrades are checked as
claims resolve. Opponents choose their best few targets with preference noise.
Active managers can also make a free pickup after claims. Week 1 is free agency.

The race excluding us records opponent markets and cut bars. Our roster/budget
variants are replayed through those same seasons to choose this week's best modeled
bid within the guide ceiling (`ranker/claims.py`). For our team only, each roster and
cash variant compares all three future spending/saving plans through the championship.
One plan is chosen by its average outcome across seasons, never separately using a
record's future prices or scores. This searches a small family of continuation
strategies, not every possible sequence of future auction decisions.
Winning and losing outcomes are evaluated per recorded season, preserving their
connection to future opportunity.
These are individual alternatives, not an optimized simultaneous claim portfolio.
Replay title odds are approximate: opponents retain players taken by our replay.
Championship weeks are scored with the roster held in each week; a Week 17 pickup
cannot improve Week 16 retroactively.
The full race uses our selected baseline saving plan when reporting league odds.
The old `room` and `hold` replay policies remain only as evaluation baselines; the
draft model's separate FAAB assumptions are unchanged.

Sleeper documents its [suggested bid ranges](https://support.sleeper.com/en/articles/12111984-suggested-faab-bids),
but its [public API](https://docs.sleeper.com/) does not document an endpoint for them.
The model does not depend on those suggestions or on future-week guide publications.

## Workflows

Refresh projections and league information, compute this week's FAAB bids, and
optimize the lineup. Run before the week's waiver deadline, and again before games
start to update the lineup. No manual input:

```bash
uv run refresh_season.py
```

It refetches projections (DraftSharks weekly for the weeks still to play, Sleeper
season), fetches the league state (including Sleeper weekly projections), and runs
the season model. The NFL week comes from Sleeper. Each stage must succeed before
the next starts; a failure exits nonzero without printing recommendations from an
older run. The season model takes about half an hour at six workers, most of it
replaying roster variants for this week's claims.

The terminal summary shows the remaining budget, recommended bids with their drops and
reserve moves, and the optimal lineup with start/sit changes. Bids are evaluated individually, so treat
them as alternatives. After this week's waivers process, recommendations are free
pickups. Enter the recommended claims and lineup on Sleeper yourself. Add `--report`
for detailed model diagnostics and league odds.

Results are saved to `league.json` and `season.json`; `uv run serve.py` displays them
at http://127.0.0.1:8123/season.html. The season model reads refreshed weekly
projections directly and uses the existing `pool.json` only to join player IDs, so
this workflow does not need a pool rebuild or a new hand-saved DraftSharks page.

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
uv run python -m unittest ranker.waiver_selftest
uv run evaluate_waivers.py > season/bidding_evaluation.json
```

Before and after changing the draft opponent model, compare
`evaluate_opponents.py`'s replay accuracy.

`evaluate_waivers.py` holds out every bid on a player before predicting that player's
positive submitted bids. It compares the original bidding formula, the guide prior,
and fitted manager behavior using mean absolute log error. It also compares the
no-reserve, balanced, and patient plans with the old room/hold policies on 2,048
paired opponent seasons using a separate seed, then repeats with every opponent
active. The report includes paired
confidence intervals and opening budget paths conditional on survival. One observed
auction cannot establish future activity or saving habits, or validate absolute
championship probabilities.
It also reconstructs the current week's pre-auction rosters and exercises paid claim
pricing with the learned model. That counterfactual uses observed bids and current
projections, so it is a diagnostic rather than an ex-ante backtest.

## Dashboards

`uv run serve.py` serves the repository at http://127.0.0.1:8123 (direct `file://`
access cannot fetch the JSON). `/` renders `rankings.json`, including the live board
state embedded in it and when that snapshot was taken; `/season.html` renders
`season.json`: this week's lineup, the waiver claims with their optimal bids, the
elimination bar by week, every team's odds and budget outlook, and the waiver market;
`/sources/` renders `data_source_matches.json` as a team-by-source fit heatmap with
pick-level evidence. Re-run `refresh_season.py` or `refresh.py` and reload to advance.
