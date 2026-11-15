"""Exception types shared by the pipeline.

Every error that should be shown to the user is raised as a ``PipelineError``
carrying a short, friendly message. Cancellation is a separate type so the
caller can tell "the user pressed Cancel" apart from a real failure.
"""

from __future__ import annotations


class PipelineError(Exception):
    """A failure with a message that is safe and useful to show in the UI."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class JobCancelled(Exception):
    """Raised inside the pipeline when the user cancels the job."""
