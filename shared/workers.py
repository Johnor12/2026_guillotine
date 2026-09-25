"""Keep simulation pools small on the shared desktop."""

from __future__ import annotations

import os

MAX_WORKERS = 8


def worker_count() -> int:
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    return max(1, min(MAX_WORKERS, cpus or 1))
