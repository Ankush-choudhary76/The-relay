"""Unit tests for Rate Limiter and Queue Manager."""

import asyncio
import pytest
from relay.queue_manager import QueueFullException, QueueManager
from relay.rate_limiter import ClientRateLimiter


def test_rate_limiter_sliding_window():
    limiter = ClientRateLimiter(rpm_limit=3, tpm_limit=500)

    # First 3 requests allowed
    assert limiter.can_proceed(100)[0] is True
    limiter.record_usage(100)
    assert limiter.can_proceed(100)[0] is True
    limiter.record_usage(100)
    assert limiter.can_proceed(100)[0] is True
    limiter.record_usage(100)

    # 4th request blocked by RPM
    can_proc, wait_s, reason = limiter.can_proceed(100)
    assert can_proc is False
    assert "RPM limit" in reason
    assert wait_s is not None and wait_s > 0


def test_rate_limiter_daily_budget():
    limiter = ClientRateLimiter(rpd_limit=2)
    limiter.record_usage(50)
    limiter.record_usage(50)

    can_proc, wait_s, reason = limiter.can_proceed(50)
    assert can_proc is False
    assert "Daily" in reason
    assert limiter.is_daily_exhausted() is True


@pytest.mark.asyncio
async def test_queue_manager_concurrency_and_overflow():
    qm = QueueManager(max_concurrent=2, max_queue_size=2)

    # Acquire 2 slots
    w1 = await qm.acquire_slot(timeout_seconds=1.0)
    w2 = await qm.acquire_slot(timeout_seconds=1.0)
    assert w1 >= 0
    assert w2 >= 0

    # Start 2 tasks that wait in queue
    async def wait_in_queue():
        return await qm.acquire_slot(timeout_seconds=2.0)

    t1 = asyncio.create_task(wait_in_queue())
    t2 = asyncio.create_task(wait_in_queue())
    await asyncio.sleep(0.05)
    assert qm.waiting_in_queue == 2

    # Next attempt should exceed max_queue_size=2 and raise QueueFullException
    with pytest.raises(QueueFullException):
        await qm.acquire_slot()

    # Release one slot to let t1 proceed
    qm.release_slot()
    w3 = await t1
    assert w3 > 0

    qm.release_slot()
    qm.release_slot()
    await t2
    qm.release_slot()

