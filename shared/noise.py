"""The weekly score noise both simulators share, and the Gaussian helpers."""

from __future__ import annotations

import math

from .league import REGULAR_WEEKS, WEEK_STARTERS, WEEKS

# SD of one team's weekly score around its expected lineup value at the 7-starter
# base shape, idiosyncratic component only: a league-wide scoring swing (weather
# slates, a dead week) moves every team together and cancels out of who gets cut, so
# it stays out of the elimination model. Expanded weeks scale by sqrt(starters/7).
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
SEED = 20260804

SIGMA_WEEK = tuple(
    WEEKLY_SIGMA * math.sqrt(WEEK_STARTERS[w] / WEEK_STARTERS[0]) for w in range(WEEKS)
)
# A championship score is two independent weeks of noise on top of two expected values.
SIGMA_CHAMP = math.hypot(SIGMA_WEEK[REGULAR_WEEKS], SIGMA_WEEK[REGULAR_WEEKS + 1])

_SQRT2 = math.sqrt(2.0)
_INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(z: float) -> float:
    return math.exp(-0.5 * z * z) * _INV_SQRT2PI


def cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / _SQRT2))
