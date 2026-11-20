"""utils: the periodic loop."""

from __future__ import annotations

import asyncio

from virag.utils import cancel, every


def test_every_runs_the_job_on_the_interval_and_survives_its_failures(caplog):
    runs = []

    async def job():
        runs.append(len(runs))

        if len(runs) == 2:
            raise RuntimeError("one bad run")

    async def go():
        task = asyncio.create_task(every(0.005, job, name="the job"))
        await asyncio.sleep(0.05)
        await cancel(task)

    asyncio.run(go())

    assert len(runs) >= 3                                               # went on after the failure
    assert "the job failed" in caplog.text and "one bad run" in caplog.text


def test_every_with_no_interval_never_runs():
    runs = []

    async def job():
        runs.append(1)

    asyncio.run(asyncio.wait_for(every(0, job, name="the job"), timeout=1))

    assert runs == []
