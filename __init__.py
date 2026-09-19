"""Football Manager modelling backend. No Streamlit dependency or bundled real save data."""

from .errors import DataError, NotFittedError
from .data import prepare_player_export, read_player_export
from .drivers import GoalDriverModel
from .league import LeagueModel
from .pipeline import MoneyballModel
from .players import PlayerValuationModel

__all__ = ["DataError", "NotFittedError", "LeagueModel", "GoalDriverModel", "PlayerValuationModel",
           "MoneyballModel", "prepare_player_export", "read_player_export"]
__version__ = "0.1.0"
