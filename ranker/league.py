"""This league's shape (from README.md) and the strategy constants.

32 teams, 0.5 PPR guillotine with a +1.0/rec TE premium, 1 QB. The two lowest
weekly scores are eliminated each of weeks 1-15 (their players hit waivers),
then the last two teams play a week 16-17 total-points championship. Starters
open at 1 QB / 1 RB / 2 WR / 1 TE / 2 W-R-T = 7, plus 1 bench = 8 draftable
roster spots = 8 rounds = 256 picks, all offense (no D/ST or K slot). The 2
reserve spots are not drafted into. Snake with a third-round reversal, and
picks can be traded. Lineups expand in-season (WEEKLY_SHAPES below); the extra
bench weeks matter only for holding players and are not modeled.

The geometry (teams, rounds, my slot, reversal, and everything derived from them) is a
default: draft.json is authoritative, and `configure_from_draft()` rebinds it before
any board is built, so test drafts with other league and roster sizes work unchanged.
The starting-lineup shape and the strategy knobs are this league's and stay fixed —
draft.json carries no lineup information.
"""

from __future__ import annotations

SCHEME = "half_ppr"
POINTS_FIELD = "points"  # season column in pool.json; weekly_points carries the value input
POSITIONS = ("QB", "RB", "WR", "TE")

# --- guillotine season structure --------------------------------------------------
# Weeks 1-15 each cut the two lowest weekly scores (30 of 32 teams); the last two
# play a week 16-17 total-points championship. Week 18 exists in the NFL but not here.
REGULAR_WEEKS = 15
WEEKS = 17
CHAMPIONSHIP_WEEKS = (16, 17)


def _week_shape(week: int) -> dict[str, int]:
    """Starting slots in a given week, from the league's expansion schedule.

    Base 1 QB / 1 RB / 2 WR / 1 TE / 2 FLEX; +1 WR at week 7, +1 RB at week 9,
    +1 FLEX at week 12, +1 superflex at week 14. The superflex is modeled as a
    second dedicated QB slot: a QB nearly always outscores the flex-caliber
    alternative and a QB waiver body is always available, so the seat's realistic
    occupant is a QB. This slightly undervalues RB/WR/TE depth in weeks 14-17.
    """
    return {
        "QB": 2 if week >= 14 else 1,
        "RB": 2 if week >= 9 else 1,
        "WR": 3 if week >= 7 else 2,
        "TE": 1,
        "FLEX": 3 if week >= 12 else 2,
    }


WEEKLY_SHAPES = tuple(_week_shape(w) for w in range(1, WEEKS + 1))
WEEK_STARTERS = tuple(sum(s.values()) for s in WEEKLY_SHAPES)

# Waiver-tier bodies a surviving roster holds by week, per position. The roster grows
# from 8 spots (week 1) to 16 (week 14) while only 8 players are ever drafted, so
# in-season adds accumulate on every surviving team; each is modeled as an
# always-available body at that week's wire level. The allocation leans RB/WR — where
# injury churn drives adds — with the second QB arriving for the week-14 superflex.
# This is what stops drafted depth from being credited with the whole late-season
# lineup: by week 15 half of everyone's roster came off the wire.
WEEK_WIRE_BODIES = {
    "QB": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2),
    "RB": (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4),
    "WR": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4),
    "TE": (1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3),
}

TEAMS = 32
MY_SLOT = 20
STARTING_SLOTS = {"QB": 1, "RB": 1, "WR": 2, "TE": 1, "FLEX": 2}
# Slots no other position can cover, so every roster must end up with at least these.
DEDICATED_SLOTS = {"QB": 1, "RB": 1, "WR": 2, "TE": 1}
# This league sets no per-position caps, so the roster size is the only bound.
MAX_POSITIONS = {"QB": 8, "RB": 8, "WR": 8, "TE": 8}
# Every roster slot is offense: the league has no D/ST or kicker slot at all.
DST_SLOTS = 0
BENCH_SLOTS = 1
IR_SLOTS = 2  # the league's reserve spots; not drafted into
ROSTER_SLOTS = sum(STARTING_SLOTS.values()) + DST_SLOTS + BENCH_SLOTS  # 8
ROUNDS = ROSTER_SLOTS
TOTAL_PICKS = TEAMS * ROUNDS  # 256
# Round the snake stops alternating: round 3 repeats round 2's direction, inverting
# parity from there on (forward, reverse, reverse, forward, reverse, forward, ...).
REVERSAL_ROUND = 3

# Most restrictive slot first: a dedicated slot is always the cheapest place to put a
# player, which is what lets the greedy lineup solver be exact; --selftest checks that
# against brute force.
SLOT_CHAIN = {
    "QB": ("QB",),
    "RB": ("RB", "FLEX"),
    "WR": ("WR", "FLEX"),
    "TE": ("TE", "FLEX"),
}
SLOT_ELIGIBLE = {
    "QB": ("QB",),
    "RB": ("RB",),
    "WR": ("WR",),
    "TE": ("TE",),
    "FLEX": ("RB", "WR", "TE"),
}


# --- dynamic geometry -------------------------------------------------------------
# Modules that consume the geometry above read it as `league.X` attributes, never
# `from .league import X`, so a rebind here reaches everyone.


def configure(teams: int, rounds: int, my_slot: int, reversal_round: int = 0) -> None:
    """Rebind the geometry to a draft's actual shape. Strategy knobs are untouched."""
    global TEAMS, MY_SLOT, ROUNDS, ROSTER_SLOTS, BENCH_SLOTS, TOTAL_PICKS, REVERSAL_ROUND
    starters = sum(STARTING_SLOTS.values()) + DST_SLOTS
    if rounds < starters:
        raise ValueError(f"{rounds} rounds cannot fill the {starters} starting slots")
    if rounds > sum(MAX_POSITIONS.values()):
        raise ValueError(f"{rounds} rounds cannot be drafted under the caps {MAX_POSITIONS}")
    if not 1 <= my_slot <= teams:
        raise ValueError(f"my slot {my_slot} is not a slot in a {teams}-team draft")
    TEAMS, ROUNDS, MY_SLOT = teams, rounds, my_slot
    ROSTER_SLOTS = rounds
    BENCH_SLOTS = rounds - starters
    TOTAL_PICKS = teams * rounds
    REVERSAL_ROUND = reversal_round


def configure_from_draft(raw: dict) -> None:
    """Adopt draft.json's geometry: format.{teams,rounds,reversal_round}, my slot."""
    fmt = raw.get("format") or {}
    if not fmt.get("teams") or not fmt.get("rounds"):
        raise ValueError("no format.teams/format.rounds to configure the league from")
    # An unpublished draft order leaves me.draft_slot null; the README default stands.
    my_slot = (raw.get("me") or {}).get("draft_slot") or MY_SLOT
    configure(
        int(fmt["teams"]),
        int(fmt["rounds"]),
        int(my_slot),
        int(fmt.get("reversal_round") or 0),
    )


# --- strategy knobs ---------------------------------------------------------------

# Chance a player is unavailable when a lineup job must be filled. The expected-lineup
# solver applies these position-wide assumptions to the whole depth chart: the weekly
# lineup is re-optimized across positions, and a body's contribution is the exact
# probability that the re-optimized lineup calls on it. Byes and known absences are
# now explicit zero weeks in weekly_points, so these rates price only the surprise
# in-week unavailability (injury, inactive, benching) — about a bye's worth (~6%)
# lower than when one season aggregate had to absorb byes too.
UNAVAILABLE_RATE = {"QB": 0.05, "RB": 0.15, "WR": 0.08, "TE": 0.06}

# --- guillotine model knobs -------------------------------------------------------
# SD of one team's weekly score around its expected lineup value at the 7-starter
# base shape, idiosyncratic component only: a league-wide scoring swing (weather
# slates, a dead week) moves every team together and cancels out of who gets cut, so
# it stays out of the elimination model. Expanded weeks scale this by sqrt(starters/7)
# in guillotine.py.
WEEKLY_SIGMA = 16.0
# Persistent per-team error in the projections themselves, as weekly-mean points: a
# team projected to average 100 truly averages 92-108 at one sigma. Drawn once per
# team per simulated season — opponents and me alike — so bad projections can survive,
# good ones can die, and my own busts are priced into every week's safety margin.
TEAM_SEASON_SIGMA = 8.0
# A full starting lineup's blowup week bottoms out around two sigma below expectation;
# the Gaussian tail below that is an artifact, and the weekly minimum over 31 teams
# otherwise lives in that artifact tail and drags the elimination bar absurdly low.
SCORE_FLOOR_Z = -2.2
# Simulated guillotine seasons per fixed-point iteration for elimination bars.
GUILLOTINE_SIMS = 512
# Tier spacing for in-season replacement on eliminated rosters: after a cut, the best
# dropped players are bid away with real FAAB by ~30 competing survivors, so a
# roster's j-th waiver body sits at the (j * WIRE_DROP_RANK)-th best dropped player at
# its position. Drives the weekly waiver-wire escalation in guillotine.py.
WIRE_DROP_RANK = 6
SURVIVAL_SIGMA = 3.5  # softness of "will he last until my next pick"
# Candidates per position considered for the next pick by the bulk policy.
LOOKAHEAD_PER_POS = 2
# The live decision gets a broader pool than the bulk policy: the top three players
# at each position, which retains useful interior tradeoffs behind each position's head.
FIRST_PICK_PER_POS = 3
# A live-board candidate or later target must survive to that decision in at least one
# redraw out of twenty. Rarer paths are noise, not useful draft choices.
CANDIDATE_SURVIVAL_FLOOR = 0.05
# The live decision plans targets across this many of my held picks before the ordinary
# two-pick policy resumes. Four reaches across both sides of the next snake turn here.
LOOKAHEAD_PICKS = 4
# An entirely unfilled dedicated starter group receives a 3x source-rank boost;
# the boost fades linearly as that position's dedicated starters are filled.
OPPONENT_BALANCE_STRENGTH = 2.0
# Opponents become increasingly reluctant to add players beyond these comfortable depths.
# The penalty starts at the 2nd QB/TE and the 4th RB/WR — past what a roster with 8
# offensive spots ordinarily carries — so it only prices the extremes, and the caps in
# MAX_POSITIONS remain the hard limits. The targets sum to the 8 roster spots: they
# describe a realizable roster, and QB urgency is priced by OPPONENT_POSITION_TILT
# below, not by loosening this profile. My slot never uses this heuristic.
OPPONENT_DEPTH_TARGETS = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
OPPONENT_DEPTH_PENALTY = 2.0
# Flat source-rank multiplier per position; < 1 pulls the position up an opponent's
# board. Every available source board is a 1QB board, but 32 teams and the week-14
# superflex make QBs far scarcer here than any of them prices: at 1.0 the simulated
# room leaves starting QBs on the board into round 7, which nobody believes. 0.2 is a
# prior, not a fit (the league has no completed picks yet): it moves the 32nd starting
# QB off the board around round 7 -> 5 and prices roughly a 5x-ADP QB urgency without
# distorting the rest of the order. Refit with evaluate_opponents.py once real picks
# exist.
OPPONENT_POSITION_TILT: dict[str, float] = {"QB": 0.2}
# Multiplier around each opponent's fitted source adherence: 1 reproduces the observed
# mean log-rank loss before roster-balance adjustments, while 0 removes random variation.
NOISE = 1.0
# Cap on fixed-point iterations before a cycle must have closed.
MAX_ITERS = 40
SIMS = 200
ROLLOUT_SIMS = 100  # full-draft playouts per candidate at my next pick (planning.rollout)
SEED = 20260804


# --- draft order ------------------------------------------------------------------


def draft_order(teams: int | None = None, rounds: int | None = None) -> list[int]:
    """Slot (1-based) picking at each overall pick, honoring the reversal round.

    Odd rounds forward, even rounds reverse; from REVERSAL_ROUND on the parity is
    inverted, so round 3 repeats round 2's direction. Pinned to the README's stated
    picks for slot 20 (1.20, 2.13, 3.13, 4.20, ..., 7.13, 8.20) in validate().
    Defaults resolve at call time so a configure() rebind is honored.
    """
    teams = TEAMS if teams is None else teams
    rounds = ROUNDS if rounds is None else rounds
    order: list[int] = []
    for rnd in range(1, rounds + 1):
        forward = rnd % 2 == 1
        if REVERSAL_ROUND and rnd >= REVERSAL_ROUND:
            forward = not forward
        order.extend(range(1, teams + 1) if forward else range(teams, 0, -1))
    return order


def pick_label(pick_no: int, teams: int | None = None) -> str:
    """1-based overall pick number -> 'round.slot-in-round' as the draft room shows it."""
    rnd, idx = divmod(pick_no - 1, TEAMS if teams is None else teams)
    return f"{rnd + 1}.{idx + 1:02d}"


def picks_for_slot(slot: int, order: list[int]) -> list[int]:
    return [i + 1 for i, s in enumerate(order) if s == slot]
