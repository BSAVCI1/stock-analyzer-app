"""Deterministic exchange-session calendar."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from functools import lru_cache
from zoneinfo import ZoneInfo

from .models import ExchangeSession


def _observed(
    holiday: date,
) -> date:
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)

    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)

    return holiday


def _nth_weekday(
    year: int,
    month: int,
    weekday: int,
    occurrence: int,
) -> date:
    current = date(year, month, 1)

    offset = (
        weekday - current.weekday()
    ) % 7

    return (
        current
        + timedelta(
            days=offset
            + 7 * (occurrence - 1)
        )
    )


def _last_weekday(
    year: int,
    month: int,
    weekday: int,
) -> date:
    if month == 12:
        current = date(
            year + 1,
            1,
            1,
        ) - timedelta(days=1)
    else:
        current = date(
            year,
            month + 1,
            1,
        ) - timedelta(days=1)

    offset = (
        current.weekday() - weekday
    ) % 7

    return current - timedelta(
        days=offset
    )


def _easter_sunday(
    year: int,
) -> date:
    """Gregorian Easter calculation."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (
        19 * a
        + b
        - d
        - g
        + 15
    ) % 30

    i = c // 4
    k = c % 4

    month_offset = (
        32
        + 2 * e
        + 2 * i
        - h
        - k
    ) % 7

    correction = (
        a
        + 11 * h
        + 22 * month_offset
    ) // 451

    month = (
        h
        + month_offset
        - 7 * correction
        + 114
    ) // 31

    day = (
        (
            h
            + month_offset
            - 7 * correction
            + 114
        )
        % 31
    ) + 1

    return date(
        year,
        month,
        day,
    )


@lru_cache(maxsize=None)
def nyse_regular_holidays(
    year: int,
) -> frozenset[date]:
    holidays: set[date] = set()

    holidays.add(
        _observed(
            date(year, 1, 1)
        )
    )

    # New Year's Day of the following year
    # can be observed on 31 December.
    next_new_year = _observed(
        date(year + 1, 1, 1)
    )

    if next_new_year.year == year:
        holidays.add(next_new_year)

    holidays.add(
        _nth_weekday(
            year,
            1,
            0,
            3,
        )
    )

    holidays.add(
        _nth_weekday(
            year,
            2,
            0,
            3,
        )
    )

    holidays.add(
        _easter_sunday(year)
        - timedelta(days=2)
    )

    holidays.add(
        _last_weekday(
            year,
            5,
            0,
        )
    )

    if year >= 2022:
        holidays.add(
            _observed(
                date(year, 6, 19)
            )
        )

    holidays.add(
        _observed(
            date(year, 7, 4)
        )
    )

    holidays.add(
        _nth_weekday(
            year,
            9,
            0,
            1,
        )
    )

    holidays.add(
        _nth_weekday(
            year,
            11,
            3,
            4,
        )
    )

    holidays.add(
        _observed(
            date(year, 12, 25)
        )
    )

    return frozenset(holidays)


@dataclass(frozen=True, slots=True)
class ExchangeCalendar:
    exchange_code: str = "XNYS"
    timezone_name: str = (
        "America/New_York"
    )

    regular_open: time = time(
        9,
        30,
    )

    regular_close: time = time(
        16,
        0,
    )

    exceptional_closures: frozenset[
        date
    ] = frozenset()

    def __post_init__(self) -> None:
        if (
            self.exchange_code
            .strip()
            .upper()
            != "XNYS"
        ):
            raise ValueError(
                "Only XNYS is currently "
                "supported."
            )

        ZoneInfo(self.timezone_name)

    @property
    def timezone(self) -> ZoneInfo:
        return ZoneInfo(
            self.timezone_name
        )

    def is_session(
        self,
        session_date: date,
    ) -> bool:
        if session_date.weekday() >= 5:
            return False

        if (
            session_date
            in nyse_regular_holidays(
                session_date.year
            )
        ):
            return False

        if (
            session_date
            in self.exceptional_closures
        ):
            return False

        return True

    def session(
        self,
        session_date: date,
    ) -> ExchangeSession | None:
        if not self.is_session(
            session_date
        ):
            return None

        opens_at = datetime.combine(
            session_date,
            self.regular_open,
            tzinfo=self.timezone,
        )

        closes_at = datetime.combine(
            session_date,
            self.regular_close,
            tzinfo=self.timezone,
        )

        return ExchangeSession(
            exchange_code=(
                self.exchange_code
            ),
            session_date=session_date,
            opens_at=opens_at,
            closes_at=closes_at,
        )

    def session_for_run(
        self,
        scheduled_for: datetime,
    ) -> ExchangeSession | None:
        if (
            scheduled_for.tzinfo is None
            or scheduled_for.utcoffset()
            is None
        ):
            raise ValueError(
                "scheduled_for must be "
                "timezone-aware."
            )

        local = scheduled_for.astimezone(
            self.timezone
        )

        return self.session(
            local.date()
        )

    def is_after_close(
        self,
        scheduled_for: datetime,
    ) -> bool:
        session = self.session_for_run(
            scheduled_for
        )

        if session is None:
            return False

        local = scheduled_for.astimezone(
            self.timezone
        )

        return local >= session.closes_at

    def is_last_session_of_week(
        self,
        session_date: date,
    ) -> bool:
        if not self.is_session(
            session_date
        ):
            return False

        iso_year, iso_week, _ = (
            session_date.isocalendar()
        )

        candidate = (
            session_date
            + timedelta(days=1)
        )

        while True:
            candidate_year, candidate_week, _ = (
                candidate.isocalendar()
            )

            if (
                candidate_year,
                candidate_week,
            ) != (
                iso_year,
                iso_week,
            ):
                return True

            if self.is_session(candidate):
                return False

            candidate += timedelta(days=1)
