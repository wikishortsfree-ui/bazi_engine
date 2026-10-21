"""
True local solar time correction for bāzì (八字) calculation.

Administrative timezone (e.g. China Standard Time) is fixed offset from UTC and does
not match true solar time at a given longitude. Two corrections are needed:

1. Longitude: each degree east/west of the timezone reference meridian shifts
   local solar noon by 4 minutes. So we apply lng × 4 minutes from UTC.

2. Equation of Time (EOT): the Sun's apparent position runs fast or slow relative
   to mean solar time (up to ~±16 minutes), so we add EOT from pvlib.

Without these, the hour pillar (時柱) can be wrong near hour boundaries, especially
at longitudes far from the zone meridian or around solstices/equinoxes when EOT
is large.

For American times: pass a timezone-aware datetime (e.g. ZoneInfo('America/New_York')),
or a naive datetime plus tz (e.g. tz='America/Los_Angeles'). DST is applied
automatically from the IANA database—no extra correction needed.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pvlib.solarposition


def to_true_solar_time(
    dt: datetime,
    lat: float,
    lng: float,
    *,
    tz: str | None = None,
) -> datetime:
    """
    Convert to true local solar time at (lat, lng).

    If dt is naive and tz is given (e.g. 'America/New_York'), the time is
    localized first so daylight saving is applied before converting to UTC.
    For American time, use tz='America/New_York', 'America/Chicago',
    'America/Denver', or 'America/Los_Angeles'.

    Then applies:
    - Longitude offset: lng × 4 minutes from UTC.
    - Equation of Time from pvlib (minutes to add).

    Returns a naive datetime representing true local solar time.
    """
    if dt.tzinfo is None:
        if tz is None:
            raise ValueError("datetime must be timezone-aware or pass tz= (e.g. tz='America/Los_Angeles')")
        from zoneinfo import ZoneInfo
        dt = dt.replace(tzinfo=ZoneInfo(tz))

    solpos = pvlib.solarposition.get_solarposition(dt, lat, lng)
    eot_minutes = float(solpos["equation_of_time"].iloc[0])

    lng_offset_minutes = lng * 4
    utc_dt = dt.astimezone(timezone.utc)
    true_solar_dt = utc_dt + timedelta(minutes=lng_offset_minutes + eot_minutes)
    return true_solar_dt.replace(tzinfo=None)
