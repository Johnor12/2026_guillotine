"""Sleeper identifiers and the one HTTP helper every fetch uses."""

from __future__ import annotations

import json
import time
import urllib.request

LEAGUE_ID = "1397662420398247936"  # Gnosis Guillotine
MY_USER_ID = "1127785716420898816"  # johnor; the id is stable where display names are not
API = "https://api.sleeper.app/v1"


def get_json(url: str, timeout: int = 60):
    # Sleeper's CDN serves snapshots up to minutes stale and can even roll a pick count
    # backwards between refreshes; a unique query param forces origin. Its hosts also
    # reject urllib's default agent with a 403.
    bust = f"{'&' if '?' in url else '?'}nocache={time.time_ns()}"
    request = urllib.request.Request(
        url + bust, headers={"Accept": "application/json", "User-Agent": "curl/8.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def score(stats: dict, scoring: dict) -> float:
    """League points for a stat line: the dot product Sleeper's players page shows."""
    return round(sum(scoring[k] * v for k, v in stats.items() if k in scoring and v), 2)
