"""Domain-specific exceptions used by MeasurePilot."""


class MeasurePilotError(Exception):
    """Base class for expected MeasurePilot failures."""


class ProjectValidationError(MeasurePilotError):
    """Raised when a project model or archive violates an invariant."""


class ProjectFormatError(ProjectValidationError):
    """Raised when a file is not a supported MeasurePilot project."""


class UnsafeArchiveError(ProjectFormatError):
    """Raised when an archive contains unsafe or unexpected members."""
