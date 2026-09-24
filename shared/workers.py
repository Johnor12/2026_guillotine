"""Keep simulation pools small on the shared desktop."""

from __future__ import annotations

import os

# Six workers are known safe on this host; preserve samples and take longer.
MAX_WORKERS = 6


def worker_count() -> int:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return max(1, min(MAX_WORKERS, cpus or 1))
