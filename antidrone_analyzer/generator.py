"""Synthetic (dummy) data generator for a counter-UAS sensor site.

Every file produced here is fake but follows the shape of real equipment
output, so the analyzer can be tested by anyone without hardware:

* ``iq_dump.bin``        raw ADC capture, int16 interleaved I/Q (little endian)
* ``iq_dump.json``       capture metadata (sample rate, center freq, scale)
* ``rf_detections.csv``  RF sensor detection log
* ``radar_tracks.csv``   radar track log
* ``system_error.log``   equipment event / error log
* ``ground_truth.json``  what was injected, used by the tests to score results
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FULL_SCALE = 32767


@dataclass
class IQConfig:
    sample_rate: float = 10e6      # complex samples per second
    center_freq: float = 2.44e9    # RF center of the 2.4 GHz ISM capture
    duration: float = 0.05         # seconds
    noise_rms: float = 300.0       # ADC counts
    seed: int = 7


# --------------------------------------------------------------------------
# Raw IQ / ADC dump
# --------------------------------------------------------------------------
def _band_noise(rng, n, fs, f_lo, f_hi):
    """Complex noise limited to [f_lo, f_hi] Hz (baseband) via FFT masking."""
    spec = rng.normal(size=n) + 1j * rng.normal(size=n)
    freqs = np.fft.fftfreq(n, 1 / fs)
    spec[(freqs < f_lo) | (freqs > f_hi)] = 0
    sig = np.fft.ifft(spec)
    return sig / (np.std(sig) + 1e-12)


def generate_iq(path: Path, cfg: IQConfig) -> dict:
    """Write an int16 I/Q dump with drone emissions and injected glitches."""
    rng = np.random.default_rng(cfg.seed)
    fs = cfg.sample_rate
    n = int(cfg.duration * fs)
    t = np.arange(n) / fs
    x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * cfg.noise_rms / np.sqrt(2)

    truth = {"emitters": [], "glitches": []}

    # 1) FHSS control link: 0.4 ms bursts every 1 ms on hopping channels
    hop_set = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]) * 1e6
    burst_len, period = 0.4e-3, 1.0e-3
    hops = []
    for start in np.arange(0.002, cfg.duration - burst_len, period):
        f = hop_set[rng.integers(len(hop_set))]
        i0, i1 = int(start * fs), int((start + burst_len) * fs)
        x[i0:i1] += 2500 * np.exp(2j * np.pi * f * t[i0:i1])
        hops.append(float(f))
    truth["emitters"].append({
        "type": "FHSS_control", "burst_ms": burst_len * 1e3, "period_ms": period * 1e3,
        "hop_channels_mhz": sorted({h / 1e6 for h in hops}),
    })

    # 2) Wideband video downlink (~1.6 MHz wide around -4.2 MHz), second half only
    v0 = int(0.5 * n)
    video = _band_noise(rng, n - v0, fs, -5.0e6, -3.4e6) * 1200
    x[v0:] += video
    truth["emitters"].append({
        "type": "video_downlink", "center_mhz": -4.2, "bandwidth_mhz": 1.6,
        "start_s": v0 / fs,
    })

    iq = np.empty(2 * n, dtype=np.float64)
    iq[0::2], iq[1::2] = x.real, x.imag

    # 3) Injected glitches (positions are in complex-sample index)
    def mark(kind, s0, s1):
        truth["glitches"].append({"type": kind, "start_s": s0 / fs, "end_s": s1 / fs})

    # ADC saturation: front-end overload by a nearby jammer pulse
    s0 = int(0.0123 * fs); s1 = s0 + 1500
    iq[2 * s0:2 * s1] *= 40
    mark("saturation", s0, s1)
    # Dropout: DMA buffer underrun -> zeros
    s0 = int(0.0251 * fs); s1 = s0 + 2048
    iq[2 * s0:2 * s1] = 0
    mark("dropout", s0, s1)
    # Single impulse spikes (ESD / switching transients)
    for tt in (0.0333, 0.0412):
        s0 = int(tt * fs)
        iq[2 * s0:2 * s0 + 6] = rng.choice([-1, 1], 6) * 30000
        mark("spike", s0, s0 + 3)
    # DC offset step (LO leakage after re-tune)
    s0 = int(0.0450 * fs); s1 = s0 + 20000
    iq[2 * s0:2 * s1:2] += 4000
    mark("dc_step", s0, s1)

    iq = np.clip(np.round(iq), -FULL_SCALE, FULL_SCALE).astype("<i2")
    iq.tofile(path)
    meta = asdict(cfg) | {"format": "int16_iq_interleaved_le", "full_scale": FULL_SCALE,
                          "n_samples": n}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return truth


# --------------------------------------------------------------------------
# Sensor logs (30-minute scenario with 3 drone sorties + clutter)
# --------------------------------------------------------------------------
SORTIES = [  # (start_min, duration_min, start_bearing, bearing_rate deg/s, protocol)
    (3.0, 6.0, 40.0, 0.05, "DJI_OcuSync"),
    (14.0, 4.0, 210.0, -0.08, "FHSS_2.4G"),
    (22.5, 5.0, 300.0, 0.03, "DJI_OcuSync"),
]


def generate_logs(out_dir: Path, start: datetime, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    rf_rows, radar_rows = [], []
    truth = {"sorties": []}

    for idx, (m0, dur, b0, rate, proto) in enumerate(SORTIES, start=1):
        t0 = start + timedelta(minutes=m0)
        n = int(dur * 60)
        truth["sorties"].append({"id": idx, "start": t0.isoformat(),
                                 "end": (t0 + timedelta(seconds=n)).isoformat(),
                                 "protocol": proto})
        rng_m = 1800.0
        for s in range(n):
            ts = t0 + timedelta(seconds=s)
            bearing = (b0 + rate * s) % 360
            rng_m = max(150.0, rng_m - rng.uniform(1, 6))
            # radar sees it every second (with a few misses)
            if rng.random() > 0.05:
                radar_rows.append([ts + timedelta(milliseconds=int(rng.integers(0, 300))),
                                   f"T{idx:03d}", round(rng_m + rng.normal(0, 8), 1),
                                   round((bearing + rng.normal(0, 1.0)) % 360, 2),
                                   round(rng.uniform(3, 15), 1), round(rng.uniform(4, 14), 1),
                                   round(rng.normal(-17, 2), 1)])
            # RF sensor reports roughly every 2 s
            if s % 2 == 0 and rng.random() > 0.1:
                f = 2412 + 5 * rng.integers(0, 13) if proto == "FHSS_2.4G" else 2440.0
                rf_rows.append([ts + timedelta(milliseconds=int(rng.integers(0, 500))),
                                "RF-01", float(f), 10.0 if proto != "FHSS_2.4G" else 1.0,
                                round(-95 + 40 * (1 - rng_m / 2000) + rng.normal(0, 2), 1),
                                proto, round((bearing + rng.normal(0, 4.0)) % 360, 1)])

    # clutter: birds on radar (low RCS, slow), Wi-Fi on RF (no radar match)
    for _ in range(140):
        ts = start + timedelta(seconds=float(rng.uniform(0, 1800)))
        radar_rows.append([ts, f"T{int(rng.integers(100, 160)):03d}", round(rng.uniform(200, 2500), 1),
                           round(rng.uniform(0, 360), 2), round(rng.uniform(5, 40), 1),
                           round(rng.uniform(2, 9), 1), round(rng.normal(-28, 3), 1)])
    for _ in range(90):
        ts = start + timedelta(seconds=float(rng.uniform(0, 1800)))
        rf_rows.append([ts, "RF-01", float(2412 + 5 * rng.integers(0, 13)), 20.0,
                        round(rng.normal(-82, 4), 1), "WiFi", round(rng.uniform(0, 360), 1)])

    rf = pd.DataFrame(rf_rows, columns=["timestamp", "sensor_id", "freq_mhz", "bandwidth_mhz",
                                        "rssi_dbm", "protocol", "bearing_deg"]).sort_values("timestamp")
    radar = pd.DataFrame(radar_rows, columns=["timestamp", "track_id", "range_m", "azimuth_deg",
                                              "elevation_m", "speed_mps", "rcs_dbsm"]).sort_values("timestamp")
    rf.to_csv(out_dir / "rf_detections.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
    radar.to_csv(out_dir / "radar_tracks.csv", index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")

    # equipment error log, with one injected error storm
    lines = []
    codes = [("INFO", "HEARTBEAT", "RF-01"), ("WARN", "GPS_WEAK", "RADAR-01"),
             ("ERROR", "CRC_ERROR", "LINK-ETH0"), ("ERROR", "ADC_OVERFLOW", "RF-01"),
             ("WARN", "TEMP_HIGH", "RADAR-01")]
    for s in range(0, 1800, 5):
        ts = start + timedelta(seconds=s, milliseconds=int(rng.integers(0, 999)))
        lvl, code, dev = codes[0]
        lines.append((ts, f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f" [{lvl}] {dev} {code} seq={s // 5}"))
        if rng.random() < 0.08:
            lvl, code, dev = codes[int(rng.integers(1, len(codes)))]
            lines.append((ts, f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f" [{lvl}] {dev} {code} value={int(rng.integers(1, 99))}"))
    storm0 = start + timedelta(minutes=17)
    for k in range(120):  # CRC storm: 2 minutes, ~1 per second
        ts = storm0 + timedelta(seconds=k, milliseconds=int(rng.integers(0, 999)))
        lines.append((ts, f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + " [ERROR] LINK-ETH0 CRC_ERROR frame_drop=1"))
    lines.append((storm0 + timedelta(seconds=30), "### corrupted line \x00\x00 ###"))  # malformed on purpose
    lines.sort(key=lambda r: r[0])
    (out_dir / "system_error.log").write_text("\n".join(text for _, text in lines) + "\n", encoding="utf-8")
    truth["error_storm"] = {"start": storm0.isoformat(),
                            "end": (storm0 + timedelta(minutes=2)).isoformat(), "code": "CRC_ERROR"}
    return truth


def generate_all(out_dir: str | Path, iq_duration: float = 0.05, seed: int = 7) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth = {"iq": generate_iq(out / "iq_dump.bin", IQConfig(duration=iq_duration, seed=seed))}
    truth["logs"] = generate_logs(out, datetime(2026, 11, 2, 9, 0, 0))
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    return out
