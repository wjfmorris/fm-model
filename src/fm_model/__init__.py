"""Public API for the FM26 Moneyball modelling backend."""

from .errors import DataError, NotFittedError
from .data import prepare_player_export, read_player_export, read_table
from .drivers import GoalDriverModel
from .league import LeagueModel
from .pipeline import MoneyballModel
from .players import PlayerValuationModel
from .outcomes import PlayerOutcomeModel
from .planning import SquadPlanner
from .metric_learning import (
    learn_metric_importance,
    metric_inventory,
    select_learned_metrics,
)
from .roles import FORMATION_PRESETS, ROLE_LABELS, canonical_role
from .catalogue import DATA_CATALOGUE

__all__ = [
    "DATA_CATALOGUE",
    "DataError",
    "NotFittedError",
    "LeagueModel",
    "GoalDriverModel",
    "PlayerValuationModel",
    "PlayerOutcomeModel",
    "SquadPlanner",
    "learn_metric_importance",
    "metric_inventory",
    "select_learned_metrics",
    "FORMATION_PRESETS",
    "ROLE_LABELS",
    "canonical_role",
    "MoneyballModel",
    "prepare_player_export",
    "read_player_export",
    "read_table",
]

__version__ = "0.3.0"
