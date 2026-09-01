"""Blend the projections with each other and with the market, position by position.

A draft that maximizes one source's numbers systematically lands on the players that
source is most wrong about (the optimizer's curse): a QB the market drafts 235th but
one projection has as QB4 becomes a "value" the ranker builds a roster around, and the
roster then grades itself with the same numbers. So a player's season points are the
equal-weight mean of DraftSharks (the weekly source), Sleeper (season, league scoring)
and the market-implied level: what the two projections together say the player at his
consensus rank within his position is worth. Rank-matching within position keeps the
league's scoring intact — the consensus boards score TE without the premium and price
QBs for one-QB rooms, but neither changes much about who the 8th TE or 20th QB is.
DraftSharks' weekly profile is scaled by the same ratio, so byes and known absences
stay zero weeks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pool import Player, by_position


def blend_to_market(players: list[Player], boards_path: Path) -> None:
    """Replace every player's points and weekly profile with the blended value, in place."""
    boards = json.loads(boards_path.read_text())
    consensus = next(s for s in boards["sources"] if s["id"] == "consensus")
    market_rank = {str(row["sleeper_id"]): row["rank"] for row in consensus["players"]}
    missing = [p.name for p in players if str(p.sleeper_id) not in market_rank]
    if missing:
        raise ValueError(f"consensus board is missing {len(missing)} pool players: {missing[:5]}")

    for group in by_position(players).values():
        projection = {p.player_id: (p.points + p.sleeper_points) / 2 for p in group}
        curve = sorted(projection.values(), reverse=True)
        by_market = sorted(group, key=lambda p: (market_rank[str(p.sleeper_id)], p.player_id))
        for k, p in enumerate(by_market):
            if p.points <= 0.0:
                continue  # no weekly profile to scale: the weekly source says he does not play
            blended = (p.points + p.sleeper_points + curve[k]) / 3
            ratio = blended / p.points
            p.weekly = tuple(round(w * ratio, 2) for w in p.weekly)
            p.points = round(sum(p.weekly), 2)
    players.sort(key=lambda p: (-p.points, p.player_id))
