"""Player-name normalization and the one name-to-Sleeper-id resolver.

No provider shares an id with any other, so joins go by name in three tiers, each
requiring exactly one survivor at the same position (an ambiguous name is left
unmatched): the full normalized name; the name without a Jr./III-style suffix; and
last name plus NFL team, for editorial first names (Cam/Cameron, Tank/Nathaniel). Team
breaks a same-name tie but never vetoes a lone candidate, since providers disagree
about who plays where in the off-season.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def words(value: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", value).casefold()
    return re.findall(r"[a-z0-9]+", plain)


def normalized_name(value: str, *, drop_suffix: bool = False) -> str:
    parts = words(value)
    if drop_suffix:
        while len(parts) > 1 and parts[-1] in SUFFIXES:
            parts.pop()
    return "".join(parts)


def base_words(value: str) -> list[str]:
    """Words with a generational suffix dropped."""
    parts = words(value)
    while len(parts) > 1 and parts[-1] in SUFFIXES:
        parts.pop()
    return parts


class Resolver:
    """Resolve (name, position, team) onto rows that carry a Sleeper id."""

    def __init__(self, rows: list[dict]):
        self.ids = {str(row["sleeper_id"]): row for row in rows}
        self.full: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.base: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.last: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            position = row["position"]
            self.full[(normalized_name(row["name"]), position)].append(row)
            self.base[(normalized_name(row["name"], drop_suffix=True), position)].append(row)
            self.last[(base_words(row["name"])[-1], position)].append(row)

    def resolve(self, name: str, position: str, team: str | None) -> tuple[dict | None, str]:
        """(matched row or None, tier label)."""
        for tier, candidates in (
            ("name", self.full.get((normalized_name(name), position), [])),
            ("suffix", self.base.get((normalized_name(name, drop_suffix=True), position), [])),
        ):
            if len(candidates) > 1 and team:
                candidates = [c for c in candidates if c.get("team") == team] or candidates
            if len(candidates) == 1:
                return candidates[0], tier
            if candidates:
                return None, f"ambiguous {tier}"
        # The first name is editorial, so the team carries the join. A row with no team
        # yet (a fresh signing) has nothing to confirm it, so it only counts when the
        # first names are prefixes (Jam/Jamarion).
        if team:
            parts = base_words(name)
            candidates = [
                c
                for c in self.last.get((parts[-1], position), [])
                if c.get("team") == team
                or (
                    not c.get("team")
                    and (parts[0].startswith(base_words(c["name"])[0])
                         or base_words(c["name"])[0].startswith(parts[0]))
                )
            ]
            if len(candidates) == 1:
                return candidates[0], "last_name_team"
            if candidates:
                return None, "ambiguous last_name_team"
        return None, "none"

    def resolve_row(self, row: dict) -> str | None:
        """A provider row's Sleeper id: the one it supplies when the pool carries it,
        else by name."""
        supplied = row.get("sleeper_id")
        if supplied is not None and str(supplied) in self.ids:
            return str(supplied)
        found, _ = self.resolve(row["name"], row["position"], row.get("team"))
        return str(found["sleeper_id"]) if found else None
