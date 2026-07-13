"""Domain-specific exceptions with user-facing configuration context."""


class PipelineError(Exception):
    """Base exception for expected pipeline failures."""


class ConfigurationError(PipelineError):
    """Raised when a configuration file cannot be loaded or composed."""


class SchemaValidationError(PipelineError):
    """Raised when parsed configuration does not satisfy a public contract."""
