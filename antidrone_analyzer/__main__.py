"""Command line interface.

    python -m antidrone_analyzer generate --out samples
    python -m antidrone_analyzer analyze  --data samples --out reports/latest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .generator import generate_all
from .iq_analysis import analyze_iq
from .log_analysis import analyze_error_log, fuse, load_detections
from .report import write_report


def _p(msg: str) -> None:
    print(f"[ADLA] {msg}", flush=True)


def cmd_generate(a) -> int:
    out = generate_all(a.out, iq_duration=a.iq_seconds, seed=a.seed)
    size = (out / "iq_dump.bin").stat().st_size / 1e6
    _p(f"sample data written to {out}/  (iq_dump.bin = {size:.1f} MB)")
    return 0


def cmd_analyze(a) -> int:
    d = Path(a.data)
    iq_path = Path(a.iq) if a.iq else d / "iq_dump.bin"
    rf_path = Path(a.rf) if a.rf else d / "rf_detections.csv"
    radar_path = Path(a.radar) if a.radar else d / "radar_tracks.csv"
    log_path = Path(a.log) if a.log else d / "system_error.log"
    inputs, iq = {}, None
    t0 = time.time()

    if iq_path.exists():
        _p(f"IQ dump   : {iq_path} ({iq_path.stat().st_size / 1e6:.1f} MB) ...")
        iq = analyze_iq(iq_path, chunk_samples=a.chunk)
        inputs["iq"] = iq_path.name
        _p(f"  -> {len(iq.glitches)} glitch intervals, {len(iq.emitter_summary)} emitters")
    fu = None
    if rf_path.exists() and radar_path.exists():
        _p(f"RF/radar  : {rf_path.name} + {radar_path.name} ...")
        fu = fuse(*load_detections(rf_path, radar_path), dt_s=a.dt, gate_deg=a.gate)
        inputs["rf"], inputs["radar"] = rf_path.name, radar_path.name
        _p(f"  -> {fu.stats['confirmed_uav_tracks']} confirmed UAV tracks")
    err = None
    if log_path.exists():
        _p(f"Error log : {log_path.name} ...")
        err = analyze_error_log(log_path)
        inputs["log"] = log_path.name
        _p(f"  -> {len(err.events)} events, {len(err.storms)} storms, {len(err.malformed_lines)} malformed lines")
    if not inputs:
        _p("no input files found - run `python -m antidrone_analyzer generate` first")
        return 2

    report = write_report(a.out, iq=iq, err=err, fu=fu, inputs=inputs)
    _p(f"done in {time.time() - t0:.1f}s -> {report}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="antidrone_analyzer",
                                 description="Anti-drone sensor log analyzer (IQ glitch/FFT, RF-radar fusion, error storms)")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create dummy sensor data")
    g.add_argument("--out", default="samples")
    g.add_argument("--iq-seconds", type=float, default=0.05, help="IQ capture length (0.05 s = 2 MB; 25 s = 1 GB)")
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_generate)

    z = sub.add_parser("analyze", help="analyze a data folder and write an HTML/CSV report")
    z.add_argument("--data", default="samples", help="folder with iq_dump.bin, rf_detections.csv, radar_tracks.csv, system_error.log")
    z.add_argument("--iq"); z.add_argument("--rf"); z.add_argument("--radar"); z.add_argument("--log")
    z.add_argument("--out", default="reports/latest")
    z.add_argument("--chunk", type=int, default=1 << 20, help="IQ samples per chunk (memory bound)")
    z.add_argument("--dt", type=float, default=1.5, help="RF-radar time gate [s]")
    z.add_argument("--gate", type=float, default=12.0, help="RF-radar bearing gate [deg]")
    z.set_defaults(func=cmd_analyze)

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
