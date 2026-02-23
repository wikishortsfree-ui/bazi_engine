from lunar_python import Solar
import pvlib
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

def to_true_solar_time(dt: datetime, lat: float, lng: float) -> datetime:
    """
    Convert a wall-clock datetime to true local solar time,
    accounting for both longitude offset AND the Equation of Time.
    """
    # Localize if naive
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    
    # pvlib needs a DatetimeIndex
    times = pd.DatetimeIndex([dt])
    
    # Get solar position — this gives us equation_of_time in minutes
    solpos = pvlib.solarposition.get_solarposition(times, lat, lng)
    eot_minutes = solpos['equation_of_time'].iloc[0]
    
    # Longitude offset: each degree = 4 minutes from UTC
    lng_offset_minutes = lng * 4
    
    # Total correction from UTC
    from datetime import timezone, timedelta
    utc_dt = dt.astimezone(timezone.utc)
    true_solar_dt = utc_dt + timedelta(minutes=lng_offset_minutes + eot_minutes)
    
    return true_solar_dt

# Test with a known chart — pick one you can verify manually
solar = Solar.fromYmdHms(2003, 4, 24, 21, 30, 0)
lunar = solar.getLunar()
ec = lunar.getEightChar()

# Four pillars
print("Year:", ec.getYear(), "Month:", ec.getMonth())
print("Day:", ec.getDay(), "Hour:", ec.getTime())

# Five elements
print("Year WuXing:", ec.getYearWuXing())

# Ten gods
print("Year ShiShen (Gan):", ec.getYearShiShenGan())
print("Year ShiShen (Zhi):", ec.getYearShiShenZhi())

# Da Yun (need gender: 1=male, 0=female)
yun = ec.getYun(1)
print("Da Yun start age:", yun.getStartYear())
for dy in yun.getDaYun():
    print("  DaYun:", dy.getGanZhi(), "start:", dy.getStartYear())