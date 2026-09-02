"""Blend the two projection sources the pool carries in this league's scoring.

A draft that maximizes one source's numbers systematically lands on the players that
source is most wrong about (the optimizer's curse), and then grades the roster with the
same numbers. DraftSharks is the thesis here — a statistical model, trusted over expert
or market opinion — so it keeps two thirds of the weight; Sleeper's season projection,
an independent model scored with the league's live settings, takes the other third and
damps the outliers the two models disagree on. No ranking or ADP board enters: they
price 1QB rooms without the TE premium, and their order is not evidence about this
league. DraftSharks' weekly profile is scaled by the blended-to-original ratio, so byes
and known absences stay zero weeks.
"""

from __future__ import annotations

from .pool import Player

DRAFTSHARKS_WEIGHT = 2 / 3


def blend_projections(players: list[Player]) -> None:
    """Replace every player's points and weekly profile with the blend, in place."""
    for p in players:
        if p.points <= 0.0:
            continue  # no weekly profile to scale: the weekly source says he does not play
        blended = DRAFTSHARKS_WEIGHT * p.points + (1 - DRAFTSHARKS_WEIGHT) * p.sleeper_points
        ratio = blended / p.points
        p.weekly = tuple(round(w * ratio, 2) for w in p.weekly)
        p.points = round(sum(p.weekly), 2)
    players.sort(key=lambda p: (-p.points, p.player_id))
