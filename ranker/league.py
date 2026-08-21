"""This league's shape (from README.md) and the strategy constants.

10 teams, 0.5 PPR redraft, 1 QB. Starters are 1 QB / 2 RB / 2 WR / 1 TE /
2 W-R-T = 8. Then 4 bench = 12 draftable roster spots = 12 rounds = 120 picks.
The 1 IR spot is not drafted into. Plain snake, no reversal.

The geometry (teams, rounds, my slot, and everything derived from them) is a default:
draft.json is authoritative, and `configure_from_draft()` rebinds it before any board
is built, so test drafts with other league and roster sizes work unchanged. The
starting-lineup shape and the strategy knobs are this league's and stay fixed —
draft.json carries no lineup information.
"""

from __future__ import annotations

SCHEME = "half_ppr"
POINTS_FIELD = "points"  # the one value column in pool.json: one-season projected points
POSITIONS = ("QB", "RB", "WR", "TE")

TEAMS = 10
MY_SLOT = 2
STARTING_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}
# Slots no other position can cover, so every roster must end up with at least these.
DEDICATED_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
# Sleeper's per-position roster caps: the draft room refuses a pick past these.
MAX_POSITIONS = {"QB": 4, "RB": 8, "WR": 8, "TE": 3}
BENCH_SLOTS = 4
IR_SLOTS = 1  # not drafted into
ROSTER_SLOTS = sum(STARTING_SLOTS.values()) + BENCH_SLOTS  # 12
ROUNDS = ROSTER_SLOTS
TOTAL_PICKS = TEAMS * ROUNDS  # 120

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


def configure(teams: int, rounds: int, my_slot: int) -> None:
    """Rebind the geometry to a draft's actual shape. Strategy knobs are untouched."""
    global TEAMS, MY_SLOT, ROUNDS, ROSTER_SLOTS, BENCH_SLOTS, TOTAL_PICKS
    starters = sum(STARTING_SLOTS.values())
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


def configure_from_draft(raw: dict) -> None:
    """Adopt draft.json's geometry: format.teams, format.rounds, and my draft slot."""
    fmt = raw.get("format") or {}
    if not fmt.get("teams") or not fmt.get("rounds"):
        raise ValueError("no format.teams/format.rounds to configure the league from")
    # An unpublished draft order leaves me.draft_slot null; the README default stands.
    my_slot = (raw.get("me") or {}).get("draft_slot") or MY_SLOT
    configure(int(fmt["teams"]), int(fmt["rounds"]), int(my_slot))


# --- strategy knobs ---------------------------------------------------------------

# Chance a player is unavailable when a lineup job must be filled. The expected-lineup
# solver applies these position-wide assumptions to the whole depth chart: the weekly
# lineup is re-optimized across positions, and a body's contribution is the exact
# probability that the re-optimized lineup calls on it.
UNAVAILABLE_RATE = {"QB": 0.08, "RB": 0.20, "WR": 0.12, "TE": 0.10}
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
# The penalty starts at the 3rd QB/TE and the 7th RB/WR — past what a 12-spot redraft
# roster ordinarily carries — so it only prices the extremes, and the position caps in
# MAX_POSITIONS remain the hard limits. My slot never uses this heuristic.
OPPONENT_DEPTH_TARGETS = {"QB": 2, "RB": 6, "WR": 6, "TE": 2}
OPPONENT_DEPTH_PENALTY = 2.0
# Flat source-rank multiplier per position; < 1 pulls the position up an opponent's
# board. The RB tilt carries over from the previous league (fitted there at 0.67):
# largely the same drafters, and they've expressed a strong preference for RBs.
# Set at 0.75 pending this league's own replay evidence (evaluate_opponents.py).
OPPONENT_POSITION_TILT: dict[str, float] = {"RB": 0.67}
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
    """Slot (1-based) picking at each overall pick. Plain snake, no reversal.

    Odd rounds forward, even rounds reverse. Pinned to the README's stated picks for
    slot 2 (1.02, 2.09, 3.02, 4.09, ..., 11.02, 12.09) in validate(). Defaults resolve
    at call time so a configure() rebind is honored.
    """
    teams = TEAMS if teams is None else teams
    rounds = ROUNDS if rounds is None else rounds
    order: list[int] = []
    for rnd in range(1, rounds + 1):
        forward = rnd % 2 == 1
        order.extend(range(1, teams + 1) if forward else range(teams, 0, -1))
    return order


def pick_label(pick_no: int, teams: int | None = None) -> str:
    """1-based overall pick number -> 'round.slot-in-round' as the draft room shows it."""
    rnd, idx = divmod(pick_no - 1, TEAMS if teams is None else teams)
    return f"{rnd + 1}.{idx + 1:02d}"


def picks_for_slot(slot: int, order: list[int]) -> list[int]:
    return [i + 1 for i, s in enumerate(order) if s == slot]
