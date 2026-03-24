"""
APScheduler — weekly scheduled scans.
Timezone defaults to TZ env var (set in docker-compose) or America/Chicago.
"""
import logging
import os
from typing import Callable, Awaitable, Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

TZ = pytz.timezone(os.environ.get("TZ", "America/Chicago"))
_scheduler: Optional[AsyncIOScheduler] = None
_scan_cb: Optional[Callable] = None


def register_scan_callback(cb: Callable):
    global _scan_cb
    _scan_cb = cb


async def _scheduled(service_type: str):
    logger.info(f"Scheduled scan: {service_type}")
    if _scan_cb:
        await _scan_cb(service_type=service_type)


def build_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler:
        return _scheduler

    _scheduler = AsyncIOScheduler(timezone=TZ)

    _scheduler.add_job(
        _scheduled,
        CronTrigger(day_of_week="sun", hour=9,  minute=45, timezone=TZ),
        id="sun_morning",
        kwargs={"service_type": "Sunday Morning"},
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled,
        CronTrigger(day_of_week="sun", hour=11, minute=30, timezone=TZ),
        id="sun_midday",
        kwargs={"service_type": "Sunday Midday"},
        replace_existing=True,
    )
    _scheduler.add_job(
        _scheduled,
        CronTrigger(day_of_week="wed", hour=19, minute=30, timezone=TZ),
        id="wed_evening",
        kwargs={"service_type": "Wednesday Evening"},
        replace_existing=True,
    )

    logger.info("Scheduler: Sun 9:45am | Sun 11:30am | Wed 7:30pm  (America/Chicago)")
    return _scheduler
