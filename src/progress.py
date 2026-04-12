from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


class Progress:
    """Small terminal progress helper for long-running evaluation scripts."""

    def __init__(
        self,
        total: int,
        label: str = "Progress",
        *,
        enabled: bool = True,
        min_interval: float = 0.5,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.enabled = enabled
        self.min_interval = min_interval
        self.count = 0
        self.started_at = time.perf_counter()
        self._last_render_at = 0.0
        self._closed = False

        if self.enabled:
            self._render(force=True)

    def advance(self, step: int = 1) -> None:
        if self._closed:
            return

        self.count += max(int(step), 0)
        if self.total:
            self.count = min(self.count, self.total)

        now = time.perf_counter()
        should_render = (
            self.count == self.total
            or now - self._last_render_at >= self.min_interval
        )
        if should_render:
            self._render(force=self.count == self.total)

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled:
            self._render(force=True)
            sys.stderr.write("\n")
            sys.stderr.flush()
        self._closed = True

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return

        now = time.perf_counter()
        if not force and now - self._last_render_at < self.min_interval:
            return

        elapsed = max(now - self.started_at, 0.0)
        if self.total:
            ratio = min(self.count / self.total, 1.0)
            percent = ratio * 100
            rate = self.count / elapsed if elapsed > 0 else 0.0
            remaining = self.total - self.count
            eta = remaining / rate if rate > 0 else 0.0
            message = (
                f"\r{self.label}: {self.count}/{self.total} "
                f"({percent:5.1f}%) elapsed {elapsed:6.1f}s ETA {eta:6.1f}s"
            )
        else:
            message = f"\r{self.label}: {self.count} elapsed {elapsed:6.1f}s"

        sys.stderr.write(message)
        sys.stderr.flush()
        self._last_render_at = now


def iter_progress(
    items: Iterable[T],
    *,
    total: int | None = None,
    label: str = "Progress",
    enabled: bool = True,
) -> Iterator[T]:
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = 0

    progress = Progress(total or 0, label, enabled=enabled)
    try:
        for item in items:
            yield item
            progress.advance()
    finally:
        progress.close()
