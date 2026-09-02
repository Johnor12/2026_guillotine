"""This league's shape (from README.md) and the strategy constants.

32 teams, 0.5 PPR guillotine with a +1.0/rec TE premium, 1 QB. The two lowest
weekly scores are eliminated each of weeks 1-15 (their players hit waivers),
then the last two teams play a week 16-17 total-points championship. Starters
open at 1 QB / 1 RB / 2 WR / 1 TE / 2 W-R-T = 7, plus 1 bench = 8 draftable
roster spots = 8 rounds = 256 picks, all offense (no D/ST or K slot). The 2
reserve spots are not drafted into. Snake with a third-round reversal, and
picks can be traded. Lineups expand in-season (WEEKLY_SHAPES below); the extra
bench weeks matter only for holding players and are not modeled. The league sets
no per-position roster caps.

These are constants, not configuration: board.load_board complains loudly when
draft.json disagrees with them, which is the cue to edit this file.
"""

from __future__ import annotations

POSITIONS = ("QB", "RB", "WR", "TE")

# --- guillotine season structure --------------------------------------------------
# Weeks 1-15 each cut the two lowest weekly scores (30 of 32 teams); the last two
# play a week 16-17 total-points championship. Week 18 exists in the NFL but not here.
REGULAR_WEEKS = 15
WEEKS = 17


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
# always-available body at that week's wire level. The allocation leans RB/WR, where
# injury churn drives adds, with the second QB arriving for the week-14 superflex.
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
BENCH_SLOTS = 1
ROUNDS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS  # 8
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


# --- strategy knobs ---------------------------------------------------------------

# Chance a player is unavailable when a lineup job must be filled. The expected-lineup
# solver applies these position-wide assumptions to the whole depth chart: the weekly
# lineup is re-optimized across positions, and a body's contribution is the exact
# probability that the re-optimized lineup calls on it. Byes and known absences are
# explicit zero weeks in weekly_points, so these rates price only the surprise
# in-week unavailability (injury, inactive, benching).
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
# team per simulated season, opponents and me alike, so bad projections can survive,
# good ones can die, and my own busts are priced into every week's safety margin.
TEAM_SEASON_SIGMA = 8.0
# A full starting lineup's blowup week bottoms out around two sigma below expectation;
# the Gaussian tail below that is an artifact, and the weekly minimum over 31 teams
# otherwise lives in that artifact tail and drags the elimination bar absurdly low.
SCORE_FLOOR_Z = -2.2
# Simulated guillotine seasons per fixed-point iteration for elimination bars.
GUILLOTINE_SIMS = 512
SURVIVAL_SIGMA = 3.5  # softness of "will he last until my next pick"
# Candidates per position considered for the next pick by the bulk policy.
LOOKAHEAD_PER_POS = 2
# The live decision gets a broader pool than the bulk policy: the top three players
# at each position, which retains useful interior tradeoffs behind each position's head.
FIRST_PICK_PER_POS = 3
# A live-board candidate or later target must survive to that decision in at least one
# redraw out of twenty. Rarer paths are noise, not useful draft choices.
CANDIDATE_SURVIVAL_FLOOR = 0.05
# Survivors of that floor go to the branch redraws and full-draft rollouts in two-pick
# score order, up to this many: the shortlist is the deterministic board's positional
# heads plus the live board's, and the tail of it never wins the rollout.
ROLLOUT_CANDIDATES = 8
# The live decision plans targets across this many of my held picks before the ordinary
# two-pick policy resumes. Four reaches across both sides of the next snake turn here.
LOOKAHEAD_PICKS = 4
# An entirely unfilled dedicated starter group receives a 3x source-rank boost;
# the boost fades linearly as that position's dedicated starters are filled. QB gets a
# scarcity rule on top (simulation.Draft.opponent_candidates): the boost acts on rank
# among the players left, which is too weak to stop a team punting the one position
# with exactly 32 week-1 starters for 32 rosters.
OPPONENT_BALANCE_STRENGTH = 2.0
# Opponents become increasingly reluctant to add players beyond these comfortable depths.
# The penalty starts at the 2nd QB/TE and the 4th RB/WR, past what a roster with 8
# offensive spots ordinarily carries, so it only prices the extremes. The targets sum
# to the 8 roster spots: they describe a realizable roster, and QB urgency is priced by
# the superflex boards in the opponents' cold-start blend, not by loosening this
# profile. My slot never uses this.
OPPONENT_DEPTH_TARGETS = {"QB": 1, "RB": 3, "WR": 3, "TE": 1}
OPPONENT_DEPTH_PENALTY = 2.0
# Flat source-rank multiplier per position; < 1 pulls the position up an opponent's
# board. QB scarcity (32 teams plus the week-14 superflex) is priced by boards, not a
# tilt: an unfitted opponent drafts off the cold_start blend and a fitted one adopts a
# superflex/2QB board when its picks say so, and that room already clears the 32nd
# starting QB around round 7 with ~8 QBs gone by the end of round 2. A QB tilt on top
# of those boards double-counts (0.2 pulled 14 QBs into the first two rounds). The
# +1.0/rec TE premium is still a tilt: only Sleeper's league-scored points board
# prices it, a sliver of the blend, while the premium lifts a 70-catch TE by ~4
# points a week, puts TE2 above WR1 in this scoring and makes TE2-TE9
# first-to-third-round players on my board; a room blind to that hands me four of
# them, so 0.6 assumes the room half sees it. A prior, not a fit (no completed picks
# yet). Refit with evaluate_opponents.py once real picks exist.
OPPONENT_POSITION_TILT: dict[str, float] = {"TE": 0.6}
# Multiplier around each opponent's fitted source adherence: 1 reproduces the observed
# mean log-rank loss before roster-balance adjustments, while 0 removes random variation.
NOISE = 1.0
# Cap on fixed-point iterations before a cycle must have closed.
MAX_ITERS = 40
SIMS = 200
ROLLOUT_SIMS = 100  # full-draft playouts per candidate at my next pick (planning.rollout)
SEED = 20260804


# --- draft order ------------------------------------------------------------------


def draft_order() -> list[int]:
    """Slot (1-based) picking at each overall pick, honoring the reversal round.

    Odd rounds forward, even rounds reverse; from REVERSAL_ROUND on the parity is
    inverted, so round 3 repeats round 2's direction. Pinned to the README's stated
    picks for slot 20 (1.20, 2.13, 3.13, 4.20, ..., 7.13, 8.20) in validate().
    """
    order: list[int] = []
    for rnd in range(1, ROUNDS + 1):
        forward = rnd % 2 == 1
        if rnd >= REVERSAL_ROUND:
            forward = not forward
        order.extend(range(1, TEAMS + 1) if forward else range(TEAMS, 0, -1))
    return order


def pick_label(pick_no: int) -> str:
    """1-based overall pick number -> 'round.slot-in-round' as the draft room shows it."""
    rnd, idx = divmod(pick_no - 1, TEAMS)
    return f"{rnd + 1}.{idx + 1:02d}"


def picks_for_slot(slot: int, order: list[int]) -> list[int]:
    return [i + 1 for i, s in enumerate(order) if s == slot]
