"""Custom exception classes for the SDBench framework."""


class SDBenchError(Exception):
    """Base class for SDBench exceptions."""


class DataIngestionError(SDBenchError):
    """Raised when data ingestion fails."""


class DecisionEngineError(SDBenchError):
    """Raised when the decision engine encounters an error."""


class EvaluationError(SDBenchError):
    """Raised when evaluation of a session fails."""
