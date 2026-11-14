"""Equipment error-log analysis and RF / radar detection fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) "
    r"\[(?P<level>[A-Z]+)\] (?P<device>\S+) (?P<code>[A-Z_]+)(?: (?P<detail>.*))?$"
)


# --------------------------------------------------------------------------
# error log
# --------------------------------------------------------------------------
@dataclass
class ErrorLogResult:
    events: pd.DataFrame
    malformed_lines: list = field(default_factory=list)
    per_minute: pd.DataFrame | None = None
    storms: pd.DataFrame | None = None
    top_codes: pd.DataFrame | None = None


def parse_error_log(path: str | Path) -> tuple[pd.DataFrame, list]:
    """Stream the log line by line (works for multi-GB files)."""
    rows, bad = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, start=1):
            m = LOG_RE.match(line.rstrip("\n"))
            if m:
                rows.append(m.groupdict())
            elif line.strip():
                bad.append({"line_no": n, "text": line.strip()[:80]})
    df = pd.DataFrame(rows, columns=["ts", "level", "device", "code", "detail"])
    df["ts"] = pd.to_datetime(df["ts"], format="%Y-%m-%d %H:%M:%S.%f")
    return df, bad


def find_error_storms(events: pd.DataFrame, k_mad: float = 5.0, min_count: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-minute ERROR counts; minutes above median + k*MAD are a 'storm'."""
    err = events[events["level"] == "ERROR"].set_index("ts")
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    idx = pd.date_range(events["ts"].min().floor("min"), events["ts"].max().ceil("min"), freq="min")
    per_min = err.resample("min").size().reindex(idx, fill_value=0).rename("errors").to_frame()
    med = per_min["errors"].median()
    mad = (per_min["errors"] - med).abs().median() * 1.4826
    thr = max(min_count, med + k_mad * max(mad, 1.0))
    per_min["threshold"] = thr
    per_min["storm"] = per_min["errors"] > thr

    storms = []
    flag = per_min["storm"].to_numpy()
    starts = np.flatnonzero(np.diff(np.concatenate(([0], flag.astype(int)))) == 1)
    ends = np.flatnonzero(np.diff(np.concatenate((flag.astype(int), [0]))) == -1)
    for s, e in zip(starts, ends):
        t0, t1 = per_min.index[s], per_min.index[e] + pd.Timedelta(minutes=1)
        win = err.loc[t0:t1]
        storms.append({"start": t0, "end": t1, "errors": int(len(win)),
                       "dominant_code": win["code"].mode().iat[0] if len(win) else "",
                       "device": win["device"].mode().iat[0] if len(win) else ""})
    return per_min, pd.DataFrame(storms)


def analyze_error_log(path: str | Path) -> ErrorLogResult:
    events, bad = parse_error_log(path)
    per_min, storms = find_error_storms(events)
    top = (events[events["level"] != "INFO"].groupby(["level", "device", "code"]).size()
           .rename("count").reset_index().sort_values("count", ascending=False))
    return ErrorLogResult(events, bad, per_min, storms, top)


# --------------------------------------------------------------------------
# RF + radar fusion
# --------------------------------------------------------------------------
def load_detections(rf_path, radar_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = pd.read_csv(rf_path, parse_dates=["timestamp"])
    radar = pd.read_csv(radar_path, parse_dates=["timestamp"])
    return rf.sort_values("timestamp").reset_index(drop=True), radar.sort_values("timestamp").reset_index(drop=True)


def angle_diff(a, b):
    """Smallest absolute difference between two bearings in degrees (handles 359 vs 1)."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


@dataclass
class FusionResult:
    rf: pd.DataFrame
    radar: pd.DataFrame
    events: pd.DataFrame
    stats: dict


def fuse(rf: pd.DataFrame, radar: pd.DataFrame, dt_s: float = 1.5, gate_deg: float = 12.0,
         min_hits: int = 5) -> FusionResult:
    """Associate each RF detection with the nearest radar plot in time + bearing.

    A radar track with at least ``min_hits`` RF associations is a confirmed UAV.
    """
    rf = rf.copy()
    radar = radar.copy()
    rt = radar["timestamp"].to_numpy("datetime64[ns]").astype("int64")
    ft = rf["timestamp"].to_numpy("datetime64[ns]").astype("int64")
    win = int(dt_s * 1e9)
    lo = np.searchsorted(rt, ft - win, side="left")
    hi = np.searchsorted(rt, ft + win, side="right")

    match_track, match_cost = [], []
    az = radar["azimuth_deg"].to_numpy()
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b <= a:
            match_track.append(None); match_cost.append(np.nan); continue
        d_ang = angle_diff(az[a:b], rf.at[i, "bearing_deg"])
        d_t = np.abs(rt[a:b] - ft[i]) / 1e9
        cost = d_ang / gate_deg + d_t / dt_s
        j = int(np.argmin(np.where(d_ang <= gate_deg, cost, np.inf)))
        if d_ang[j] <= gate_deg:
            match_track.append(radar.at[a + j, "track_id"]); match_cost.append(float(cost[j]))
        else:
            match_track.append(None); match_cost.append(np.nan)
    rf["radar_track"] = match_track
    rf["match_cost"] = match_cost

    hits = rf.dropna(subset=["radar_track"]).groupby("radar_track").size()
    confirmed = set(hits[hits >= min_hits].index)
    rf["class"] = np.where(rf["radar_track"].isin(confirmed), "UAV_confirmed",
                           np.where(rf["radar_track"].notna(), "RF_radar_weak", "RF_only"))
    radar["class"] = np.where(radar["track_id"].isin(confirmed), "UAV_confirmed", "radar_only")

    events = []
    for tid in sorted(confirmed):
        r = radar[radar["track_id"] == tid]
        f = rf[rf["radar_track"] == tid]
        events.append({
            "track_id": tid, "start": r["timestamp"].min(), "end": r["timestamp"].max(),
            "duration_s": round((r["timestamp"].max() - r["timestamp"].min()).total_seconds(), 1),
            "protocol": f["protocol"].mode().iat[0], "rf_hits": len(f), "radar_plots": len(r),
            "min_range_m": round(r["range_m"].min(), 1), "mean_rcs_dbsm": round(r["rcs_dbsm"].mean(), 1),
            "max_rssi_dbm": round(f["rssi_dbm"].max(), 1),
            "mean_bearing_err_deg": round(float(angle_diff(f["bearing_deg"],
                                          r.set_index("timestamp")["azimuth_deg"].reindex(
                                              f["timestamp"], method="nearest").to_numpy()).mean()), 2),
        })
    ev = pd.DataFrame(events)
    stats = {
        "rf_detections": len(rf), "radar_plots": len(radar),
        "rf_associated": int(rf["radar_track"].notna().sum()),
        "rf_only": int((rf["class"] == "RF_only").sum()),
        "radar_only_plots": int((radar["class"] == "radar_only").sum()),
        "confirmed_uav_tracks": len(confirmed),
        "rf_false_alarm_rejected_pct": round(100 * (rf["class"] != "UAV_confirmed").mean(), 1),
        "radar_clutter_rejected_pct": round(100 * (radar["class"] == "radar_only").mean(), 1),
    }
    return FusionResult(rf, radar, ev, stats)
