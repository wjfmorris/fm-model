"""Public API for the FM26 Moneyball modelling backend."""

from .errors import DataError, NotFittedError
from .data import prepare_player_export, read_player_export, read_table
from .drivers import GoalDriverModel
from .league import LeagueModel
from .pipeline import MoneyballModel
from .players import PlayerValuationModel
from .catalogue import DATA_CATALOGUE

__all__ = [
    "DATA_CATALOGUE",
    "DataError",
    "NotFittedError",
    "LeagueModel",
    "GoalDriverModel",
    "PlayerValuationModel",
    "MoneyballModel",
    "prepare_player_export",
    "read_player_export",
    "read_table",
]

__version__ = "0.1.0"
