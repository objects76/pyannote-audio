"""Compatibility shim - re-exports from core.model for checkpoint unpickling."""

from pyannote.audio.core.model import (
    Problem,
    Resolution,
    Specifications,
    UnknownSpecificationsError,
)

# Dummy Task class for checkpoint compat (PyanNet constructor accepts task=None)
Task = type(None)

__all__ = ["Problem", "Resolution", "Specifications", "UnknownSpecificationsError", "Task"]
