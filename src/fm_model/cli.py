from __future__ import annotations

import argparse
import json

from .data import read_player_export, read_table
from .pipeline import MoneyballModel


def main(argv=None):
    parser = argparse.ArgumentParser(description="FM26 Moneyball model backend")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit", help="fit available model layers")
    fit.add_argument("--league-table", required=True)
    fit.add_argument("--team-data")
    fit.add_argument("--player-data")
    fit.add_argument("--player-league", help="Competition to attach when the player export omits it")
    fit.add_argument("--player-season", type=int, help="Save season to attach when the player export omits it")
    fit.add_argument("--player-owned", choices=("true", "false"),
                     help="Ownership flag to attach when the player export omits it")
    fit.add_argument("--out", required=True)
    fit.add_argument("--include-all-available", action="store_true")
    scenario = sub.add_parser("scenario", help="print a finishing target scenario")
    scenario.add_argument("--model", required=True)
    scenario.add_argument("--league", required=True)
    scenario.add_argument("--position", required=True, type=int)
    scenario.add_argument("--probability", type=float, default=0.7)
    args = parser.parse_args(argv)

    if args.command == "fit":
        model = MoneyballModel()
        model.fit_league(read_table(args.league_table))
        if args.team_data:
            model.fit_drivers(read_table(args.team_data), include_all_available=args.include_all_available)
        if args.player_data:
            if args.player_league or args.player_season is not None or args.player_owned is not None:
                owned = None if args.player_owned is None else args.player_owned == "true"
                players = read_player_export(args.player_data, league=args.player_league,
                                             season=args.player_season, owned=owned)
            else:
                players = read_table(args.player_data)
            model.fit_players(players)
        model.save(args.out)
        print(json.dumps(model.readiness(), indent=2))
    else:
        model = MoneyballModel.load(args.model)
        print(json.dumps(model.target_scenario(args.league, args.position, probability=args.probability), indent=2, default=float))


if __name__ == "__main__":
    main()
