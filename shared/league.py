"""This league's shape, as both the draft and the in-season model see it.

32 teams, 0.5 PPR guillotine with a +1.0/rec TE premium, 1 QB. The two lowest weekly
scores are eliminated each of weeks 1-15 (their players hit waivers), then the last
two teams play a week 16-17 total-points championship. Starters open at 1 QB / 1 RB /
2 WR / 1 TE / 2 W-R-T and expand in-season (WEEKLY_SHAPES); the bench grows with them.
No D/ST or K slot, no per-position roster caps, two reserve spots outside the counts.

These are constants, not configuration. The draft board loader and the season state
loader complain loudly when Sleeper disagrees with them, which is the cue to edit here.
"""

from __future__ import annotations

POSITIONS = ("QB", "RB", "WR", "TE")
TEAMS = 32

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
STARTING_SLOTS = {"QB": 1, "RB": 1, "WR": 2, "TE": 1, "FLEX": 2}
# Slots no other position can cover, so every roster must end up with at least these.
DEDICATED_SLOTS = {"QB": 1, "RB": 1, "WR": 2, "TE": 1}
# Most restrictive slot first: a dedicated slot is always the cheapest place to put a
# player, which is what lets the greedy lineup solver be exact.
SLOT_CHAIN = {
    "QB": ("QB",),
    "RB": ("RB", "FLEX"),
    "WR": ("WR", "FLEX"),
    "TE": ("TE", "FLEX"),
}


# The bench grows with the starting lineup, one spot at each expansion (weeks 7, 9, 12,
# 14), from 1 to 5: rosters run 8 spots in week 1 to 16 from week 14. The league states
# the endpoints; the intermediate steps are an assumption checked against Sleeper's
# roster_positions each week.
def _bench(week: int) -> int:
    return 1 + (week >= 7) + (week >= 9) + (week >= 12) + (week >= 14)


WEEK_ROSTER_SIZE = tuple(WEEK_STARTERS[w] + _bench(w + 1) for w in range(WEEKS))

FAAB_BUDGET = 1000
# Sleeper's reserve_allow_out setting: the two reserve slots hold IR, PUP and Out players.
# The model keeps a body there while his projection is zero; once it resumes he needs a
# regular spot, whatever label Sleeper still shows.
RESERVE_SLOTS = 2
RESERVE_STATUSES = frozenset({"Out", "IR", "PUP"})
# Sleeper's waiver_clear_days. After the weekly run, a dropped player sits on waivers and
# claims on him process when he clears: observed about 23 hours after the drop, on
# Sleeper's next 20-minute processing tick (07:06 -> 06:15, 12:54 -> 11:55 next day).
WAIVER_CLEAR_DAYS = 1
WAIVER_CLEAR_HOURS = 23

# Projection blend, draft and season alike: DraftSharks is the thesis (a statistical
# model, trusted over expert or market opinion) and keeps two thirds; Sleeper's
# independent league-scored projection damps the outliers the two disagree on, so a
# roster is not built around one model's misses.
DRAFTSHARKS_WEIGHT = 2 / 3
