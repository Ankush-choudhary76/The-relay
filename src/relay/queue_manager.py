"""Bounded queue manager and concurrency controller."""

import asyncio
import time
from typing import Optional
from relay.config import settings


class QueueFullException(Exception):
    pass


class QueueManager:
    def __init__(
        self,
        max_concurrent: Optional[int] = None,
        max_queue_size: Optional[int] = None,
    ):
        self.max_concurrent = max_concurrent or settings.max_concurrent_requests
        self.max_queue_size = max_queue_size or settings.queue_max_size

        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.waiting_in_queue = 0
        self.peak_queue_length = 0
        self.rejected_requests_count = 0
        self._lock = asyncio.Lock()

    async def acquire_slot(self, timeout_seconds: Optional[float] = None) -> float:
        """Enqueues for a concurrency slot.
        Returns: queue_wait_time_ms
        Raises: QueueFullException if queue is saturated, or TimeoutError if deadline expires.
        """
        t_enter = time.perf_counter()

        async with self._lock:
            if self.waiting_in_queue >= self.max_queue_size:
                self.rejected_requests_count += 1
                raise QueueFullException(
                    f"Queue saturated: {self.waiting_in_queue}/{self.max_queue_size} pending requests."
                )
            self.waiting_in_queue += 1
            if self.waiting_in_queue > self.peak_queue_length:
                self.peak_queue_length = self.waiting_in_queue

        try:
            if timeout_seconds and timeout_seconds > 0:
                await asyncio.wait_for(self.semaphore.acquire(), timeout=timeout_seconds)
            else:
                await self.semaphore.acquire()

            t_acquired = time.perf_counter()
            wait_ms = (t_acquired - t_enter) * 1000.0
            return wait_ms
        finally:
            async with self._lock:
                self.waiting_in_queue = max(0, self.waiting_in_queue - 1)

    def release_slot(self):
        """Releases concurrency slot back to semaphore."""
        self.semaphore.release()

    def get_stats(self) -> dict:
        return {
            "current_queue_size": self.waiting_in_queue,
            "peak_queue_length": self.peak_queue_length,
            "rejected_requests": self.rejected_requests_count,
            "max_concurrent": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
        }

