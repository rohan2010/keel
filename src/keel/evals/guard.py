"""Re-export for the task definitions: `tasks.py` is hash-frozen (the Chapter 6 result was measured on it) and imports
the send detector as `from .guard import looks_like_send`. The detector itself lives in `keel.agent.guard`."""

from ..agent.guard import looks_like_send

__all__ = ["looks_like_send"]
