"""Cat 7 — Session & Time Context (9 features) — v2.0.

Per Project Spec 30min §7.2 Cat 7 trim 15→9. Cyclic encodings replace v1.0
raw integer time features; intra-5m granularity features dropped (no analog
at 30m where minute is always 0 or 30).

9 features (locked per spec):
  - hour_of_day_sin               — sin(2π × hour / 24)
  - hour_of_day_cos               — cos(2π × hour / 24)
  - day_of_week_sin               — sin(2π × dayofweek / 7)
  - day_of_week_cos               — cos(2π × dayofweek / 7)
  - is_weekend                    — 1 if dayofweek ≥ 5 (Sat/Sun)
  - session_overlap_asian_london  — 1 if hour ∈ [7, 9) UTC
                                    (Tokyo × London active window)
  - session_overlap_london_ny     — 1 if hour ∈ [13, 16) UTC
                                    (London × NY active window)
  - session_overlap_ny_asian      — 1 if hour ∈ [21, 22) ∪ [0, 6) UTC
                                    (NY × Sydney/Tokyo overnight window;
                                    "Asian" = Tokyo + Sydney per TA convention)
  - month_of_year                 — integer 1..12 (categorical; spec "(1)"
                                    constrains to single feature — sin+cos
                                    would exceed count; LightGBM handles
                                    categorical-like ints natively)

DROPPED from v1.0:
  Individual session flags (session_sydney/_tokyo/_london/_new_york — joint
  info in overlaps), 4th overlap (sydney_tokyo, merged into Asian def),
  active_session_count (derivable), minutes_into_session +
  minutes_to_session_close (intra-5m granularity, 0 or 30 at 30m bars),
  session_range_vs_avg (low-importance), prev_session_range_pct (Cat 11
  territory), raw day_of_week (replaced by sin/cos), is_monday.

§7.5 TAGGING: All 9 Cat 7 features = static (fixed at bar boundary; never
mutate intrabar — hour/day/month are properties of the bar timestamp).

INPUTS: df with timestamp column OR DatetimeIndex. Function auto-detects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Cat 7 = 9 session/time features.

    Parameters
    ----------
    df : DataFrame. Must have either:
         - DatetimeIndex (preferred — used for tz-aware timestamps), OR
         - 'timestamp' column (ms since epoch, will be parsed to UTC).

    Returns
    -------
    DataFrame of 9 columns indexed like df.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        raise ValueError(
            "session_features requires either DatetimeIndex or 'timestamp' column."
        )

    # Extract time components as numpy arrays (cleaner arithmetic)
    if isinstance(ts, pd.DatetimeIndex):
        hour = ts.hour
        dow = ts.dayofweek
        month = ts.month
    else:
        # ts is a Series here (from timestamp column path)
        hour = ts.dt.hour.values
        dow = ts.dt.dayofweek.values
        month = ts.dt.month.values

    # Cyclic encodings (4 features)
    hour_of_day_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_of_day_cos = np.cos(2 * np.pi * hour / 24.0)
    day_of_week_sin = np.sin(2 * np.pi * dow / 7.0)
    day_of_week_cos = np.cos(2 * np.pi * dow / 7.0)

    # Weekend flag (1 feature) — Saturday=5, Sunday=6
    is_weekend = (dow >= 5).astype(int)

    # Session overlap flags (3 features) — UTC hour-based
    # Asian = Tokyo (00:00-09:00) + Sydney (21:00-06:00)
    # London = 07:00-16:00
    # NY = 13:00-22:00
    session_overlap_asian_london = ((hour >= 7) & (hour < 9)).astype(int)
    session_overlap_london_ny = ((hour >= 13) & (hour < 16)).astype(int)
    # NY × Asian = NY-Sydney evening overlap (21-22) + Sydney-Tokyo overnight (0-6)
    session_overlap_ny_asian = (
        ((hour >= 21) & (hour < 22)) | ((hour >= 0) & (hour < 6))
    ).astype(int)

    # Month of year (1 feature) — integer 1..12, categorical encoding
    month_of_year = month.astype(int) if hasattr(month, "astype") else month

    return pd.DataFrame(
        {
            "hour_of_day_sin": hour_of_day_sin,
            "hour_of_day_cos": hour_of_day_cos,
            "day_of_week_sin": day_of_week_sin,
            "day_of_week_cos": day_of_week_cos,
            "is_weekend": is_weekend,
            "session_overlap_asian_london": session_overlap_asian_london,
            "session_overlap_london_ny": session_overlap_london_ny,
            "session_overlap_ny_asian": session_overlap_ny_asian,
            "month_of_year": month_of_year,
        },
        index=df.index,
    )
