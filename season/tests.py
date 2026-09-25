"""Small economic and auction regressions, independent of live league files.

    uv run -m unittest season.tests
"""
import datetime as dt
import random
import statistics
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shared.league import WEEKS
from shared.noise import SIGMA_WEEK, cdf

from .claims import _record_values, claims, title_objective
from .fetch_league import processing_week
from .race import Market, RaceInputs, _auction, claim_plan, fit_roster, race_inputs, replay, simulate
from .run import league_odds, market
from .state import draftsharks_by_sleeper, lineup_points, waiver_clears
from .waivers import (ROOM_CURRENT_WEEK_WEIGHT, Bidding, Manager, PriceCurve, calibration, fit_managers, fit_price_curve,
                      off_cycle_share, spending_allowance, submitted_bids)


class WaiverTests(unittest.TestCase):
    def setUp(self):
        # Full opening roster: one QB, one RB, three WRs, three TEs.
        self.positions = [0, 1, 2, 2, 2, 3, 3, 3, 1, 2]
        self.points = [20., 15., 12., 11., 5., 14., 10., 9., 8., 13.]
        self.weekly = [self.points[:] for _ in range(WEEKS)]
        self.bidding = self.build()
        self.roster = list(range(8))

    def build(self, weekly=None, ir_until=None, current_weight=1.0):
        weekly = self.weekly if weekly is None else weekly
        return Bidding(self.positions, weekly, weekly, ir_until or [0] * 10, current_weight)

    def inputs(self, weekly=None, market=()):
        """Race inputs on a one-team league; `weekly` rebuilds the bidding on other
        projections, `market` is the records my future bids are shaded against."""
        weekly = self.weekly if weekly is None else weekly
        bidding = self.bidding if weekly is self.weekly else self.build(weekly)
        return RaceInputs(1, self.positions, weekly, weekly, [self.roster[:]], [1000], [True], 0, [8, 9], False,
                          bidding, [Manager()], {}, [1000], market=Market(list(market)))

    @staticmethod
    def record(week, outcomes):
        """A recorded opponent season whose only auction is `week`'s, at `outcomes`."""
        auctions = [[] for _ in range(WEEKS)]
        auctions[week] = [(list(outcomes), list(outcomes.values()))]
        return {"seed": 1, "my_bias": 0., "bars": [0.] * WEEKS, "scores": [[]] * WEEKS, "forecast_bars": [0.] * WEEKS,
                "alive": [2] * WEEKS, "champ_bar": 0., "runner_up_bar": 0., "auctions": auctions, "champion": 0,
                "tenures": {}}

    def test_buy_before_expansion(self):
        early = self.bidding.offers(self.roster, [8], 1000, 1)
        preparing = self.bidding.offers(self.roster, [8], 1000, 6)
        self.assertTrue(early, "Value after a distant expansion must count from week 2")
        self.assertTrue(preparing, "A cheap RB should count before the second RB slot opens")
        self.assertGreater(preparing[0].ceiling, 0)

    def test_superflex_coverage_beyond_four_weeks(self):
        self.positions[8] = 0
        for points in self.weekly:
            points[8] = 18.
        offer = self.build().offers(self.roster, [8], 1000, 1)[0]
        self.assertGreater(offer.gain, 0)
        self.assertEqual(lineup_points(self.roster + [8], self.weekly[1], self.positions, 1),
                         lineup_points(self.roster, self.weekly[1], self.positions, 1))

    def test_saving_release_and_emergency(self):
        patient = spending_allowance(1000, 1000, 1, 8, "patient", 0)
        balanced = spending_allowance(1000, 1000, 1, 8, "balanced", 0)
        self.assertLess(patient, balanced)
        self.assertLess(balanced, 1000)
        self.assertEqual(spending_allowance(317, 1000, 1, 16, "patient", 0), 317)
        self.assertEqual(spending_allowance(1000, 1000, 1, 8, "patient", .5), 1000)

    def test_opponent_noise_cannot_spend_protected_cash(self):
        inputs = self.inputs()
        inputs.me = -1
        rosters, budgets = [self.roster[:4]], [1000]
        inputs.price_curve = PriceCurve(intercept=20.)  # bids far beyond any allowance
        _auction(inputs, 1, rosters, budgets, [True], {8, 9}, random.Random(7), None, -100, ["patient"])
        self.assertGreaterEqual(budgets[0], 1000 - spending_allowance(1000, 1000, 1, 1, "patient", 0))

    def test_byes_and_legal_drops(self):
        self.weekly[2][1] = 0.0
        offer = self.build().offers(self.roster, [8], 1000, 1)[0]
        self.assertGreater(offer.ceiling, 0)
        self.assertNotIn(offer.drop, (0, 1, 2, 3), "Keep scarce starters instead of dropping by raw points")

    def test_owned_players_do_not_crowd_out_bargains(self):
        inputs = self.inputs()
        plan, allowance = claim_plan(inputs, inputs.bidding, self.roster, list(range(10)), 1000, 1, 0, random.Random(1))
        self.assertTrue(any(o.player == 9 for _, o in plan))
        self.assertTrue(all(o.player not in self.roster for _, o in plan))
        self.assertEqual(allowance, max(b for b, _ in plan))

    def test_drop_is_chosen_with_the_pickup(self):
        # Week 7's ten-man roster. QB0 misses weeks 7-9 and QB1 covers them; RB8 covers
        # RB2's five zero weeks for a point a week, so alone he is the cheapest cut.
        positions = [0, 0, 1, 1, 2, 2, 2, 3, 1, 3, 0]
        weekly = [[20., 12., 15., 14., 12., 11., 10., 14., 1., 5., 15.] for _ in range(WEEKS)]
        for w in (6, 7, 8):
            weekly[w][0] = 0.
        for w in range(9, 14):
            weekly[w][2] = 0.
        bidding = Bidding(positions, weekly, weekly, [0] * 11)
        roster = tuple(range(10))
        self.assertEqual(bidding.context(roster, 6).drops[0], 8, "Alone, the point-a-week RB is the cheapest cut")
        offer = bidding.offers(list(roster), [10], 1000, 6)[0]
        self.assertEqual(offer.drop, 1, "With a better QB arriving, the backup QB is the drop")

    def test_one_week_fix_pays_for_the_drop(self):
        # QB0 sits out week 2 only; the streamer scores once. RB7 starts every week (RB1's
        # ten zero weeks, then the expanded lineup), so the streamer is worth a claim only
        # while that cover is worth less than one week.
        def offers(cover):
            positions = [0, 1, 1, 2, 2, 2, 3, 1, 0, 0]
            weekly = [[20., 15., 14., 12., 11., 10., 14., cover, 0., 0.] for _ in range(WEEKS)]
            weekly[1][0] = 0.
            weekly[1][8] = 15.
            for w in range(2, 12):
                weekly[w][1] = 0.
            return Bidding(positions, weekly, weekly, [0] * 10).offers(list(range(8)), [8], 1000, 1)
        cheap = offers(0.5)
        self.assertEqual(cheap[0].drop, 7)
        self.assertAlmostEqual(cheap[0].gain, (15. - 0.5 * (WEEKS - 2)) / (WEEKS - 1))
        self.assertFalse(offers(2.), "Thirty points of cover outweigh one fifteen-point week")

    def test_untaken_wire_refills_the_cover(self):
        # The same thirty points of cover, but an equal RB sits untaken on the wire: hold
        # the streamer his one week, then refill the spot.
        positions = [0, 1, 1, 2, 2, 2, 3, 1, 0, 1]
        weekly = [[20., 15., 14., 12., 11., 10., 14., 2., 0., 2.] for _ in range(WEEKS)]
        weekly[1][0] = 0.
        weekly[1][8] = 15.
        for w in range(2, 12):
            weekly[w][1] = 0.
        bidding = Bidding(positions, weekly, weekly, [0] * 10)
        refilled = bidding.objective(bidding.weights, [9])
        offer = refilled.offers(list(range(8)), [8], 1000, 1)[0]
        self.assertEqual(offer.drop, 7)
        self.assertAlmostEqual(offer.gain, 15. / (WEEKS - 1))
        self.assertEqual(refilled.crunch(list(range(8)) + [8], 1), bidding.crunch(list(range(8)) + [8], 1),
                         "A cut leaves no spot to refill")

    def test_returning_starter_is_not_a_placeholder(self):
        # RB7 is out through week index 3, then starts; RB8 fills in until then.
        positions = [0, 1, 1, 2, 2, 2, 3, 1, 1]
        weekly = [[20., 15., 14., 12., 11., 10., 14., 12., 0.] for _ in range(WEEKS)]
        for w in range(1, 4):
            weekly[w][7], weekly[w][8] = 0., 13.
        self.assertFalse(Bidding(positions, weekly, weekly, [0] * 9).offers(list(range(8)), [8], 1000, 1))
        # A one-point RB on the wire does not stand in for him.
        positions.append(1)
        for points in weekly:
            points.append(1.)
        bidding = Bidding(positions, weekly, weekly, [0] * 10)
        self.assertFalse(bidding.objective(bidding.weights, [9]).offers(list(range(8)), [8], 1000, 1))

    def test_guide_rank_ignores_missed_weeks(self):
        # RB0 misses half the season but outscores the 12- and 13-point RBs whenever he plays.
        positions = [1] * 8
        weekly = [[0. if w % 2 else 15., 12.] + [13.] * 6 for w in range(WEEKS)]
        bidding = Bidding(positions, weekly, weekly, [0] * 8)
        self.assertGreater(bidding.shares[1][0], bidding.shares[1][1])

    def test_reserve_bodies_hold_no_spot_until_they_return(self):
        bidding = self.build(ir_until=[2] + [0] * 9)
        roster = list(range(8))  # eight bodies, the QB on reserve through week index 1
        self.assertIsNone(bidding.offers(roster, [9], 1000, 0)[0].drop, "Reserve leaves an open spot")
        self.assertFalse(bidding.fits(roster + [9], 8, 0))
        self.assertNotEqual(bidding.crunch(roster + [8, 9], 0), 0, "Cutting a reserve body frees nothing")
        returned = roster + [9]
        fit_roster(bidding, returned, 2)
        self.assertEqual(len(returned), 8, "The returning QB needs a regular spot")
        self.assertIn(0, returned)

    def test_draftsharks_rows_outside_the_pool_join_by_name(self):
        pool = {"players": [{"player_id": 1, "sleeper_id": "11"}]}
        weekly = {"players": [{"player_id": 1, "name": "Josh Allen", "position": "QB", "weeks": {"3": {"points": 20.}}},
                              {"player_id": 2, "name": "Marcus Mariota", "position": "QB", "weeks": {"3": {"points": 15.}}},
                              {"player_id": 3, "name": "Mike Williams", "position": "WR", "weeks": {"3": {"points": 9.}}}]}
        directory = {"11": {"name": "Josh Allen", "position": "QB"},
                     "22": {"name": "Marcus Mariota", "position": "QB"},
                     "33": {"name": "Mike Williams", "position": "WR"},
                     "34": {"name": "Mike Williams", "position": "WR"}}
        joined = draftsharks_by_sleeper(pool, weekly, directory)
        self.assertEqual(joined, {"11": {"3": 20.}, "22": {"3": 15.}}, "Ambiguous names stay unjoined")

    def test_title_leverage_ignores_safe_weeks(self):
        inputs = self.inputs()
        bars = [-1000.] * WEEKS
        bars[2] = lineup_points(self.roster, self.points, self.positions, 2)
        record = {**self.record(1, {}), "bars": bars, "auctions": [[] for _ in range(WEEKS)]}
        leverage = replay([record], inputs, self.roster, 1000, 0.4)["leverage"]
        self.assertEqual(leverage[0], 0., "A week cleared in every season carries no weight")
        self.assertGreater(leverage[1], 0.)

    def test_title_objective_keeps_what_replays_better(self):
        inputs = self.inputs()
        auctions = [[] for _ in range(WEEKS)]
        auctions[1] = [([8, 9], [5, -1])]
        for weighted_title, chosen in ((.2, "title_weighted"), (.05, "points")):
            def runs(inputs, records, variants):
                flat = set(inputs.my_bidding.weights) == {1.}
                return [{"p_title": .1 if flat else weighted_title, "leverage": [2.] + [1.] * (WEEKS - 2)}] * 3
            with patch("season.claims.run_replays", side_effect=runs):
                mine, summary = title_objective(inputs, [{"auctions": auctions, "tenures": {}, "champion": 0}])
            self.assertEqual(summary["chosen"], chosen)
            self.assertEqual({r for col in mine.my_bidding.refills[1] for r in col}, {9})
            self.assertIs(mine.bidding, inputs.bidding, "The room keeps its own objective")
        weights = [row["weight"] for row in summary["title_weights"]]
        self.assertAlmostEqual(statistics.fmean(weights), 1., places=2)
        self.assertGreater(weights[0], weights[1])
        self.assertEqual(set(mine.my_bidding.weights), {1.}, "A weighting that replays worse is not used")

    def test_final_dollars_have_no_salvage_value(self):
        offer = self.bidding.offers(self.roster, [9], 317, WEEKS - 1)[0]
        self.assertEqual(offer.ceiling, 317)
        inputs = self.inputs(market=[self.record(WEEKS - 1, {9: o}) for o in (-1, 40, 120, 500)])
        plan, _ = claim_plan(inputs, inputs.my_bidding, self.roster, [9], 317, WEEKS - 1, 0., random.Random(1), price=0.1)
        self.assertEqual(plan[0][0], 317, "Cash is worthless after the final: beat every price it can")

    def test_my_bid_is_shaded_against_the_recorded_market(self):
        # Ten records: untaken twice, a free pickup once, then prices 5, 5, 20, 40, 100, 100, 300.
        prices = [-1, -1, -2, 5, 5, 20, 40, 100, 100, 300]
        market = Market([self.record(1, {9: o}) for o in prices])
        self.assertEqual([market.best_bid(1, 9, v) for v in (2, 8, 30, 120, 250, 500, 2000)], [0, 1, 6, 21, 41, 101, 101])
        self.assertEqual(market.best_bid(1, 9, float("inf")), 301)
        self.assertEqual(market.best_bid(1, 8, 100.), 0, "Never in play: nobody to beat")
        inputs = self.inputs(market=[self.record(1, {9: o}) for o in prices])
        plan, allowance = claim_plan(inputs, inputs.my_bidding, self.roster, [9], 1000, 1, 0.9, random.Random(1), price=0.4)
        (bid, offer), = plan
        value = 0.4 * offer.gain * 1000 / (WEEKS - 1)
        self.assertEqual(bid, market.best_bid(1, 9, value), "Cut risk and the guide play no part")
        self.assertEqual(allowance, bid)
        self.assertLess(bid, value)

    def test_capacity_and_same_drop_claims(self):
        for roster in (self.roster[:], self.roster[:-1]):
            inputs = self.inputs()
            rosters, budgets, free = [roster], [1000], {8, 9}
            _auction(inputs, 1, rosters, budgets, [True], free, random.Random(7), None, 40, ["value"])
            self.assertLessEqual(len(rosters[0]), 8)
            self.assertEqual(len(rosters[0]), len(set(rosters[0])))
            self.assertFalse(set(rosters[0]) & free)
            self.assertGreaterEqual(budgets[0], 0)

    def test_losing_bid_is_activity_and_no_permanent_sleepers(self):
        state = SimpleNamespace(teams=[SimpleNamespace(roster_id=1), SimpleNamespace(roster_id=2)])
        obs = [{"team": 1, "player": 9, "week": 2, "bid": 100, "reference": 50.}]
        _, (active, quiet) = fit_managers(state, obs)
        self.assertGreater(active.activity, quiet.activity)
        self.assertGreater(quiet.activity, 0)
        self.assertGreater(quiet.participation(13, 1), quiet.activity)

    def test_room_price_curve(self):
        obs = [{"bid": round(5 * r ** 0.6), "reference": r} for r in (1., 10., 100., 1000.)]
        curve = fit_price_curve(obs)
        self.assertAlmostEqual(curve.slope, 0.6, places=2)
        self.assertAlmostEqual(curve.price(1.), 5., delta=0.2)
        self.assertEqual(fit_price_curve(obs[:1]), PriceCurve(), "Nothing to fit: the guide itself")
        self.assertEqual(PriceCurve().price(40.), 40.)

    def test_latest_duplicate_bid_and_pending_not_training_data(self):
        def tx(bid, created, status="failed"):
            return {"type": "waiver", "status": status, "week": 2, "roster_id": 1,
                    "adds": {"p": 1}, "bid": bid, "created": created}
        state = SimpleNamespace(transactions=[tx(10, "a"), tx(20, "b"), tx(30, "c", "pending")])
        self.assertEqual([t["bid"] for t in submitted_bids(state)], [20])

    def test_paired_interpolation_keeps_record_identity(self):
        self.assertEqual(_record_values([(0, [0., 1.]), (100, [1., 0.])], 25), [0.25, 0.75])

    def test_holdout_does_not_learn_its_own_player(self):
        state = SimpleNamespace(teams=[SimpleNamespace(roster_id=1)],
                                players=[SimpleNamespace(name=str(j)) for j in range(10)])
        row = {"team": 1, "week": 2, "budget": 1000, "gain": 1.}
        obs = [{**row, "player": 9, "bid": 999, "reference": 1.}] + [
            {**row, "player": 8, "bid": r, "reference": float(r)} for r in (10, 100, 1000)]
        predicted = {p["player"]: p["predicted"] for p in calibration(state, obs)["predictions"]}
        self.assertEqual(predicted["9"], 1, "Player 9's own $999 must not move his prediction")

    def test_processing_date_overrides_submission_leg(self):
        week = processing_week(dt.datetime.fromisoformat("2026-09-16T07:05:00+00:00"), dt.date(2026, 9, 9))
        self.assertEqual(week, 2)

    def test_final_week_add_cannot_improve_previous_week(self):
        expected = lineup_points(self.roster, self.points, self.positions, 15)
        expected += lineup_points(self.roster + [9], self.points, self.positions, 16)
        record = {**self.record(WEEKS - 1, {9: -1}), "champ_bar": expected}
        inputs = self.inputs(market=[record])
        result = replay([record], inputs, self.roster, 1000, 0.4)
        self.assertAlmostEqual(result["p_title"], 0.5)

    def test_pending_claim_bid_follows_title_odds_past_the_guide_ceiling(self):
        for points in self.weekly:
            points[9] = 25.
        inputs = self.inputs(self.weekly[:])
        inputs.free_agents = [9]
        state = SimpleNamespace(me=0, my_team=SimpleNamespace(roster=self.roster, faab_left=1000, reserve=[]),
                                players=[SimpleNamespace(sleeper_id=str(j), name=str(j), position="WR",
                                                         team="T", injury_status=None, source="test") for j in range(10)])
        records = []
        for price in (300, 400):
            auctions = [[] for _ in range(WEEKS)]
            auctions[1] = [([9], [price])]
            records.append({"auctions": auctions, "forecast_bars": [0.] * WEEKS})

        def values(inputs, records, variants):
            out = []
            for roster, budget, price in variants:
                title = budget * 0.00001 + (0.008 if 9 in roster else 0.)
                out.append({"p_title": title, "title_by_record": [title, title], "p_reach_final": title,
                            "p_cut_now": 0., "p_alive_by_week": [1.] * 14, "budget_by_week": [budget] * 16,
                            "budget_after_claims": [budget] * 16})
            return out

        with patch("season.claims.run_replays", side_effect=values):
            result = claims(state, inputs, records)
        candidate = result["candidates"][0]
        guide = max(o.ceiling for o in inputs.my_bidding.swaps(self.roster, 9, 1000, 1, 0., 3))
        self.assertLess(guide, 401)
        self.assertEqual(candidate["optimal_bid"], 401, "Outbid both clearing prices, past the room's guide reference")
        self.assertEqual(candidate["break_even_bid"], 800)

    def test_over_capacity_roster_is_fit_before_claims(self):
        # Nine bodies for eight spots, as when a reserve body loses Out/IR/PUP status.
        roster = self.roster + [8]
        for points in self.weekly:
            points[9] = 25.
        inputs = self.inputs(self.weekly[:])
        inputs.rosters, inputs.free_agents, inputs.waivers_ran = [roster[:]], [9], True
        state = SimpleNamespace(me=0, my_team=SimpleNamespace(roster=roster, faab_left=1000, reserve=[8]),
                                players=[SimpleNamespace(sleeper_id=str(j), name=str(j), position="WR",
                                                         team="T", injury_status=None, source="test") for j in range(10)])
        records = [{"auctions": [[] for _ in range(WEEKS)], "forecast_bars": [0.] * WEEKS}] * 2

        def values(inputs, records, variants):
            return [{"p_title": (t := 0.01 + (0.004 if 9 in r else 0.)), "title_by_record": [t, t], "p_reach_final": t,
                     "p_cut_now": 0., "p_alive_by_week": [1.] * 14, "budget_by_week": [b] * 16,
                     "budget_after_claims": [b] * 16} for r, b, _ in variants]

        with patch("season.claims.run_replays", side_effect=values):
            result = claims(state, inputs, records)
        self.assertEqual([c["name"] for c in result["forced_cuts"]], ["4"], "The cheapest bench body goes first")
        self.assertEqual(result["activate"], ["8"])
        self.assertEqual([c["name"] for c in result["candidates"]], ["9"], "A healthy pickup still fits after the cut")

    def test_reporting_excludes_eliminated_teams_and_uses_opening_cash(self):
        teams = [SimpleNamespace(roster_id=j + 1, name=str(j), username=None, is_mine=j == 0,
                                 alive=j < 2, faab_left=cash, points_for=0, points_this_week=0,
                                 roster=[], unknown=[])
                 for j, cash in enumerate((400, 200, 1000))]
        state = SimpleNamespace(teams=teams, players=[], transactions=[])
        inputs = SimpleNamespace(week0=8, budgets=[400, 200, 1000], positions=[], weekly=[[]] * WEEKS,
                                 managers=[Manager()] * 3, market_fit={})
        opening = [[400, 200, 1000]] + [[300 - k * 10, 150 - k * 5, 1000] for k in range(8)]
        after = opening[1:] + [[220, 110, 1000]]
        record = {"cut_week": [None, 12, None], "budget_path": opening, "budget_after_claims": after,
                  "champion": 0, "claims": [[] for _ in range(WEEKS)], "bars": [0.] * WEEKS, "champ_bar": 0.}
        rows = league_odds(state, inputs, [record])
        self.assertEqual(rows[0]["budget_entering_week9"], 400)
        self.assertEqual(rows[0]["spend_now"], 100)
        self.assertEqual(rows[2]["p_reach_final"], 0)
        self.assertTrue(all(p == 0 for p in rows[2]["p_alive_by_week"]))
        self.assertIsNone(rows[2]["budget_entering_week13"])
        budgets = market(state, inputs, [record], [record])["mean_budget_by_week"]
        self.assertEqual(budgets[0]["budget"], 300)
        self.assertEqual(budgets[4]["budget"], round((270 + 135) / 2))
        self.assertEqual(budgets[5]["budget"], 260, "The team cut in week 13 is absent in week 14")

    def test_opening_cash_restores_only_completed_current_week_bids(self):
        players = [SimpleNamespace(index=j, sleeper_id=str(j), name=str(j), ir_until=0,
                                   position=("QB", "RB", "WR", "TE")[pos], weekly=(points,) * WEEKS)
                   for j, (pos, points) in enumerate(zip(self.positions, self.points))]
        team = SimpleNamespace(roster_id=1, faab_left=850, roster=self.roster[:], reserve=[],
                               alive=True, is_mine=True)
        def tx(week, bid, status, created):
            return {"week": week, "bid": bid, "status": status, "created": created,
                    "type": "waiver", "roster_id": 1, "adds": {"9": 1}, "drops": {}}
        state = SimpleNamespace(week=2, players=players, teams=[team], me=0, my_team=team, free_agents=[8, 9],
                                 waivers_ran=True, on_waivers={}, transactions=[tx(1, 50, "complete", "a"),
                                                               tx(2, 100, "complete", "b"),
                                                               tx(2, 300, "failed", "c")])
        inputs = race_inputs(state)
        self.assertEqual(inputs.budgets, [850])
        self.assertEqual(inputs.opening_budgets, [950])
        self.assertEqual(inputs.bidding.current_weight, ROOM_CURRENT_WEEK_WEIGHT)
        self.assertEqual(inputs.my_bidding.current_weight, 1., "My bidding keeps its own objective")

    def test_players_dropped_since_the_run_need_a_claim(self):
        def drop(sid, at):
            return {"status": "complete", "processed_at": at, "drops": {sid: 1}}
        txs = [drop("a", "2026-09-22T07:00:00+00:00"), drop("a", "2026-09-23T07:05:00+00:00"),
               drop("b", "2026-09-22T12:00:00+00:00"), drop("c", "2026-09-23T12:01:00+00:00")]
        self.assertEqual(waiver_clears(txs, "2026-09-23T16:56:00+00:00"),
                         {"a": "2026-09-24T06:05+00:00", "c": "2026-09-24T11:01+00:00"},
                         "The latest drop restarts the clock; an older one has cleared")

    def test_off_cycle_auction_plays_only_players_on_waivers(self):
        inputs = self.inputs()
        candidates, _, _, _ = _auction(inputs, 1, [self.roster[:]], [1000], [True], {8, 9}, random.Random(7),
                                       None, 40, ["value"], pool=[9])
        self.assertEqual(candidates, [9])
        inputs.me = -1
        candidates, winning, wins, dropped = _auction(inputs, 1, [self.roster[:4]], [1000], [True], {8, 9},
                                                      random.Random(7), None, 40, ["value"], pool=[9], attention=0.)
        self.assertEqual((winning, wins, dropped), ([-1], [], []), "An inattentive room neither claims nor picks up")

    def test_auction_returns_the_drops_its_wins_make(self):
        # My claim on 9 beats the recorded $10; the $0 claim on 8 names the same drop and
        # loses, and with 9 aboard he no longer improves the roster as a free pickup.
        inputs = self.inputs(market=[self.record(1, {9: 10})])
        rosters, free = [self.roster[:]], {8, 9}
        candidates, winning, wins, dropped = _auction(inputs, 1, rosters, [1000], [True], free,
                                                      random.Random(7), None, 40, ["value"])
        self.assertEqual((candidates, winning, wins), ([9, 8], [11, -1], [(9, 0, 11)]))
        self.assertEqual(dropped, [4], "The receiver 9 displaces sits on waivers")
        self.assertEqual(free, {4, 8})

    def test_cascade_rounds_bid_on_the_previous_rounds_drops(self):
        inputs = self.inputs()
        inputs.rosters, inputs.budgets, inputs.alive = [self.roster[:] for _ in range(3)], [1000] * 3, [True] * 3
        inputs.managers, inputs.opening_budgets, inputs.week0, inputs.off_cycle_share = [Manager()] * 3, [1000] * 3, 14, .25
        rounds = []

        def auction(inputs, w, rosters, budgets, alive, free, rng, skip, bar, plans, pool=None, attention=1.):
            rounds.append((w, pool, attention))
            if w == 14 and pool is None:
                return [9], [3], [(9, 1, 3)], [4]  # the run: team 1 buys 9 and drops 4
            if w == 14 and pool == [4]:
                return [4], [-2], [(4, 2, 0)], [7]  # the cascade: team 2 picks 4 up free and drops 7
            return list(pool or []), [-1] * len(pool or []), [], []

        with patch("season.race._auction", side_effect=auction):
            rec = simulate(inputs, 1, exclude_me=False)
        self.assertEqual(rounds[:4], [(14, None, 1.), (14, [4], .25), (14, [7], .25), (15, None, 1.)],
                         "Two cascade rounds a week, at mid-week attention")
        self.assertEqual(rec["auctions"][14], [([9], [3]), ([4], [-2]), ([7], [-1])])
        self.assertEqual(rec["claims"][14], [(9, 1, 3), (4, 2, 0)])
        self.assertEqual(rec["auctions"][15], [([], [])], "No drops, no cascade")

        rounds.clear()
        inputs.waivers_ran, inputs.on_waivers = True, {5: "later"}
        with patch("season.race._auction", side_effect=auction):
            simulate(inputs, 1, exclude_me=False)
        self.assertEqual(rounds[0], (14, [5], .25), "After the run, the players still on waivers open the week")
        self.assertEqual(rounds[1][0], 15)

    def test_replay_bids_in_the_cascade_round(self):
        # 9 is dropped mid-week in the record and nobody takes him; my agent should.
        for points in self.weekly:
            points[9] = 13.
        mine = self.roster[:7] + [9]
        bars = [-1000.] * WEEKS
        bars[3] = lineup_points(mine, self.weekly[3], self.positions, 3)
        run_only = {**self.record(2, {8: 5}), "bars": bars}
        cascade = {**run_only, "auctions": [*run_only["auctions"][:2], [([8], [5]), ([9], [-1])], *run_only["auctions"][3:]]}
        inputs = self.inputs(self.weekly[:], market=[cascade])
        without = replay([run_only], inputs, self.roster, 1000, 0.4)["p_alive_by_week"]
        with_cascade = replay([cascade], inputs, self.roster, 1000, 0.4)["p_alive_by_week"]
        self.assertAlmostEqual(with_cascade[2], .5)
        self.assertLess(without[2], with_cascade[2])

    def test_off_cycle_share_pools_completed_weeks(self):
        def claim(week, team, at):
            return {"type": "waiver", "status": "failed", "week": week, "roster_id": team, "processed_at": at}
        run2, run3 = "2026-09-16T07:06:02+00:00", "2026-09-23T07:05:12+00:00"
        txs = [claim(2, 2, run2), claim(2, 3, run2), claim(2, 4, run2), claim(2, 1, run2),
               claim(2, 2, "2026-09-17T06:15:30+00:00"), claim(2, 1, "2026-09-17T06:15:30+00:00"),
               claim(3, 2, run3), claim(3, 3, "2026-09-24T06:15:30+00:00")]
        state = SimpleNamespace(week=3, transactions=txs, my_team=SimpleNamespace(roster_id=1))
        self.assertAlmostEqual(off_cycle_share(state), 1 / 3, msg="Mine and the unfinished week are excluded")

    def test_refill_pool_waits_for_the_weekly_auction(self):
        inputs = self.inputs()
        inputs.waivers_ran = True
        auctions = [[] for _ in range(WEEKS)]
        auctions[1] = [([9], [5])]  # off-cycle: 9 is claimed
        auctions[2] = [([8, 9], [-1, -1])]
        runs = [{"p_title": .1, "leverage": [1.] * (WEEKS - 1)}] * 3
        with patch("season.claims.run_replays", return_value=runs):
            mine, _ = title_objective(inputs, [{"auctions": auctions, "tenures": {}, "champion": 0}])
        self.assertEqual({r for col in mine.my_bidding.refills[1] for r in col}, {8})

    def test_replay_chooses_among_the_best_drops(self):
        options = self.bidding.swaps(self.roster, 9, 1000, 1, 0., 3)
        self.assertEqual(options[0], self.bidding.offers(self.roster, [9], 1000, 1)[0])
        self.assertEqual(len({o.drop for o in options}), len(options))
        self.assertEqual([o.gain for o in options], sorted((o.gain for o in options), reverse=True))
        runner_up = options[1].drop
        inputs = self.inputs()
        state = SimpleNamespace(me=0, my_team=SimpleNamespace(roster=self.roster, faab_left=1000, reserve=[]),
                                players=[SimpleNamespace(sleeper_id=str(j), name=str(j), position="WR",
                                                         team="T", injury_status=None, source="test") for j in range(10)])

        def values(inputs, records, variants):
            return [{"p_title": (t := .01 + (.004 if 9 in r and runner_up not in r else 0.)), "title_by_record": [t, t],
                     "p_reach_final": t, "p_cut_now": 0., "p_alive_by_week": [1.] * 14,
                     "budget_by_week": [b] * 16, "budget_after_claims": [b] * 16} for r, b, _ in variants]

        with patch("season.claims.run_replays", side_effect=values):
            result = claims(state, inputs, [{"auctions": [[] for _ in range(WEEKS)], "forecast_bars": [0.] * WEEKS}] * 2)
        self.assertEqual(result["candidates"][0]["drop"]["name"], str(runner_up))

    def test_room_values_the_week_it_bids_for(self):
        weekly = [self.points[:] for _ in range(WEEKS)]
        for v, points in enumerate(weekly):
            points[8] = 30. if v == 1 else 0.  # a one-week fill-in
            points[9] = 0. if v <= 1 else 13.  # a body who helps from next week on

        def favorite(weight):
            offers = Bidding(self.positions, weekly, weekly, [0] * 10, weight).offers(self.roster, [8, 9], 1000, 1)
            return max(offers, key=lambda o: o.gain).player

        self.assertEqual(favorite(1.), 9)
        self.assertEqual(favorite(ROOM_CURRENT_WEEK_WEIGHT), 8)

    def test_replay_budget_is_before_claims(self):
        for points in self.weekly:
            points[9] = 25.
        record = self.record(16, {9: 1})
        inputs = self.inputs(self.weekly[:], market=[record])
        inputs.week0 = 15
        result = replay([record], inputs, self.roster, 317, 0.4)
        self.assertEqual(result["budget_by_week"], [1000, 317])
        self.assertEqual(result["budget_after_claims"], [317, 315], "The final claim beats the recorded price by a dollar")

    def test_denial_charges_the_survivor_what_my_claim_takes(self):
        # The record's survivor bought 9 at week 3 and his final lineup loses 10 a week
        # without him; my claim at week 3 takes 9 from him, but never more than his margin.
        for points in self.weekly:
            points[9] = 25.
        base = {**self.record(2, {9: 3}), "champ_bar": 200., "runner_up_bar": 185.}
        record = {**base, "tenures": {9: [(0, 2, [0.] * 13 + [10., 10.])]}}
        inputs = self.inputs(self.weekly[:], market=[record])
        self.assertAlmostEqual(inputs.market.denial[2, 9], 15., msg="Capped by the margin over the runner-up")
        def title(rec):
            return replay([rec], inputs, self.roster, 1000, 0.7)["p_title"]
        self.assertGreater(title(record), title(base), "Denying the finalist lowers his bar")
        self.assertGreater(title({**record, "runner_up_bar": 100.}), title(record), "A wider margin lets the whole loss count")

    def test_taking_a_player_lowers_the_bar_his_buyer_set(self):
        # Opponent 1 bought 9 at week 3 and his lineup loses 10 a week without him. He set
        # week 4's bar five points over my lineup; with 9 mine, that bar is his score less
        # ten. In week 3 he was far from the bottom, so that bar stands.
        for points in self.weekly:
            points[9] = 25.
        mine = self.roster[:7] + [9]
        mu = lineup_points(mine, self.weekly[3], self.positions, 3)
        record = {**self.record(2, {}), "bars": [-1000.] * WEEKS, "runner_up_bar": 100.}
        record["scores"] = [[] for _ in range(WEEKS)]
        record["scores"][2], record["bars"][2] = [(mu - 5., 7), (mu, 8), (mu + 50., 1)], mu
        record["scores"][3], record["bars"][3] = [(mu - 5., 7), (mu + 5., 1), (mu + 50., 8)], mu + 5.
        held = {**record, "tenures": {9: [(1, 2, [10., 10.])]}}
        inputs = self.inputs(self.weekly[:], market=[record])
        untouched = replay([record], inputs, mine, 1000, 0.7)["p_alive_by_week"]
        lowered = replay([held], inputs, mine, 1000, 0.7)["p_alive_by_week"]
        self.assertEqual(lowered[1], untouched[1], "A buyer far from the cut moves no bar")
        self.assertGreater(lowered[2], untouched[2])
        self.assertAlmostEqual(lowered[2] / lowered[1], cdf(5. / SIGMA_WEEK[3]))
        self.assertEqual(replay([held], inputs, self.roster, 1000, 0.7)["p_alive_by_week"],
                         replay([record], inputs, self.roster, 1000, 0.7)["p_alive_by_week"],
                         "Without the player, the buyer keeps his score")

    def test_replay_budget_conditions_on_survival(self):
        for points in self.weekly:
            points[9] = 100.
        records = [{**self.record(15, {9: 1}), "bars": [-1000.] * WEEKS},  # survives, and buys 9 at week 16
                   {**self.record(15, {}), "bars": [1000.] * WEEKS}]  # cut at once, cash untouched
        inputs = self.inputs(self.weekly[:], market=records)
        inputs.week0 = 14
        result = replay(records, inputs, self.roster, 1000, 0.4)
        self.assertEqual(result["p_reach_final"], .5)
        self.assertEqual(result["budget_by_week"][-1], 998,
                         "Cash retained in a season where we were cut is not a survivor's budget")


if __name__ == "__main__":
    unittest.main()
