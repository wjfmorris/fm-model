class DataError(ValueError):
    """Actionable invalid or insufficient input data."""


class NotFittedError(RuntimeError):
    """A model needs training before this operation."""

