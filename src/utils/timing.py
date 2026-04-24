from __future__ import annotations

import time


class Timer:
    """Context-manager wall-clock timer.

    Usage:
        with Timer() as t:
            do_work()
        print(t.elapsed)  # seconds as float
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
