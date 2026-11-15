"""ClipForge processing pipeline.

This package has no dependency on the web layer: ``runner.run_job`` takes a
job folder, a source and options, and reports progress through a callback, so
it can be driven from FastAPI, a CLI, or a worker on another machine.
"""

from .errors import JobCancelled, PipelineError
from .runner import JobOptions, Source, run_job

__all__ = ["JobCancelled", "JobOptions", "PipelineError", "Source", "run_job"]
