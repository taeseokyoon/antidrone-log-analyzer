"""End-to-end tests: generate dummy data with known faults, analyze, score."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from antidrone_analyzer.__main__ import main
from antidrone_analyzer.generator import generate_all
from antidrone_analyzer.iq_analysis import analyze_iq, detect_glitches
from antidrone_analyzer.log_analysis import analyze_error_log, angle_diff, fuse, load_detections


@pytest.fixture(scope="module")
def data(tmp_path_factory) -> Path:
    return generate_all(tmp_path_factory.mktemp("samples"))


@pytest.fixture(scope="module")
def truth(data) -> dict:
    return json.loads((data / "ground_truth.json").read_text())


def _overlap(a0, a1, b0, b1):
    return min(a1, b1) > max(a0, b0)


# ---------------------------------------------------------------- IQ dump
def test_every_injected_glitch_is_found_with_right_type(data, truth):
    res = analyze_iq(data / "iq_dump.bin")
    fs = res.meta.sample_rate
    found = [(g.kind, g.start / fs, g.end / fs) for g in res.glitches]
    for g in truth["iq"]["glitches"]:
        hits = [f for f in found if f[0] == g["type"] and _overlap(f[1], f[2], g["start_s"], g["end_s"])]
        assert hits, f"missed {g}"
    assert len(found) == len(truth["iq"]["glitches"]), f"false alarms: {found}"


def test_fhss_hop_set_and_timing_recovered(data, truth):
    res = analyze_iq(data / "iq_dump.bin")
    fh = next(e for e in res.emitter_summary if e["emitter"].startswith("FHSS"))
    t = truth["iq"]["emitters"][0]
    got = sorted(float(c) for c in fh["channels_mhz"].split(","))
    assert got == t["hop_channels_mhz"]
    assert fh["period_ms"] == pytest.approx(t["period_ms"], abs=0.05)
    assert fh["burst_ms"] == pytest.approx(t["burst_ms"], abs=0.05)


def test_video_downlink_center_and_bandwidth(data, truth):
    res = analyze_iq(data / "iq_dump.bin")
    v = next(e for e in res.emitter_summary if e["emitter"].startswith("Wideband"))
    t = truth["iq"]["emitters"][1]
    assert v["center_mhz"] == pytest.approx(t["center_mhz"], abs=0.2)
    assert v["bandwidth_khz"] == pytest.approx(t["bandwidth_mhz"] * 1e3, rel=0.2)


def test_chunk_size_does_not_change_result(data):
    a = analyze_iq(data / "iq_dump.bin", chunk_samples=1 << 20)
    b = analyze_iq(data / "iq_dump.bin", chunk_samples=50_000)  # many chunk edges
    assert [g.kind for g in a.glitches] == [g.kind for g in b.glitches]


def test_clean_noise_has_no_glitches():
    rng = np.random.default_rng(0)
    raw = np.round(rng.normal(0, 300, 2 * 200_000)).astype("<i2")
    assert detect_glitches(raw, 0, 32767, dc_noise_std=300.0) == []


# ---------------------------------------------------------------- logs
def test_error_storm_and_malformed_line(data, truth):
    r = analyze_error_log(data / "system_error.log")
    assert len(r.storms) == 1
    s, t = r.storms.iloc[0], truth["logs"]["error_storm"]
    assert s["dominant_code"] == t["code"]
    assert s["start"] <= pd.Timestamp(t["start"]) < s["end"]
    assert len(r.malformed_lines) == 1


def test_fusion_confirms_exactly_the_three_sorties(data, truth):
    fu = fuse(*load_detections(data / "rf_detections.csv", data / "radar_tracks.csv"))
    assert fu.stats["confirmed_uav_tracks"] == 3
    for ev, s in zip(fu.events.sort_values("start").itertuples(), truth["logs"]["sorties"]):
        assert abs((ev.start - pd.Timestamp(s["start"])).total_seconds()) < 5
        assert ev.protocol == s["protocol"]


def test_angle_wraparound():
    assert angle_diff(359.0, 1.0) == pytest.approx(2.0)
    assert angle_diff(10.0, 350.0) == pytest.approx(20.0)


# ---------------------------------------------------------------- CLI
def test_cli_writes_report(data, tmp_path):
    assert main(["analyze", "--data", str(data), "--out", str(tmp_path)]) == 0
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Anti-Drone Sensor Log Analysis Report" in html and "data:image/png;base64" in html
    for f in ("iq_glitches.csv", "uav_events.csv", "error_storms.csv", "summary.json"):
        assert (tmp_path / f).exists()
