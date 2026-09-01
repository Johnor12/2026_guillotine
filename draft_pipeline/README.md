# Draft pipeline

`fetch_draft.py` is the network-only entry point that publishes `draft.json` on demand.
It does not cache responses because a stale live board is worse than another small fetch.
Sleeper's draft API is public and real-time, so a fetch between picks is the whole
live-draft story — there is no manual overlay step.

```bash
uv run draft_pipeline/fetch_draft.py
uv run draft_pipeline/fetch_draft.py --report
uv run draft_pipeline/fetch_draft.py --selftest
uv run draft_pipeline/fetch_draft.py --draft-id 1400304132081893376   # e.g. a league mock
```

## Internal boundaries

- `fetch_draft.py`: Sleeper requests and CLI orchestration
- `draft_board.py`: draft geometry, ownership, pick rows, and JSON document construction
- `report.py`: integrity diagnostics and the optional `pool.json` join report
- `selftest.py`: offline formats, reversal, trade, autopick, and malformed-board checks
- `paths.py`: this process's input/output locations

## Inputs and output

The fetch reads four Sleeper endpoints: the draft, made picks, traded picks, and league
users. Draft, picks, and trades are load-bearing; user lookup may fail with only names
becoming null. The default draft id is the league's; `--draft-id` points a run at
another draft, such as a league mock (a mock keeps its league in `metadata.league_id`,
so the user list still resolves). My picks are marked by my Sleeper user id, which is
stable where display names are not.

`draft.json.picks` always contains all 256 picks, indexed by gap-free `pick_no`.
Made rows carry Sleeper's player fields. Pending rows carry null player fields but already
identify the current owner, so consumers can answer who picks next and when my next pick
occurs. `sleeper_id` joins made picks to `pool.json`; Sleeper's name, position, and NFL
team are informational.

## Derived pending picks

Sleeper reports slot and roster only after a pick is made. Pending picks are derived from
the draft settings:

- normal snake rounds alternate direction;
- a reversal round repeats the prior round's direction and flips parity thereafter;
- traded ownership is applied by round and the pick's original roster.

This league is a 32-team snake with a third-round reversal: round 1 forward, rounds 2
and 3 reversed, then alternating. Slot 20 therefore owns 1.20, 2.13, 3.13, 4.20, 5.13,
6.20, 7.13, 8.20 before trades — and this league can trade picks, so `traded_picks`
matters.

Every fetch compares derived slot and roster against every made pick Sleeper reports.
Disagreements are warnings and are retained in `board_derivation`. The offline self-test
covers later rounds and trade cases the current live board may not yet exercise.
