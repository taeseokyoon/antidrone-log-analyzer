"""Figures (PNG), tables (CSV/JSON) and a self-contained HTML report."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .iq_analysis import read_chunk  # noqa: E402

GLITCH_COLORS = {"saturation": "#d62728", "dropout": "#7f7f7f", "spike": "#ff7f0e", "dc_step": "#9467bd"}
CLASS_COLORS = {"UAV_confirmed": "#d62728", "radar_only": "#9e9e9e", "RF_only": "#1f77b4",
                "RF_radar_weak": "#17becf"}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_iq_overview(iq, path: Path, max_points: int = 200_000):
    fs = iq.meta.sample_rate
    n = min(iq.n_samples, max_points * 8)
    step = max(1, n // max_points)
    i = read_chunk(iq.bin_path, 0, n)[0::2 * step].astype(float)
    t = np.arange(len(i)) * step / fs * 1e3
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, i, lw=0.4, color="#1f4e79")
    for g in iq.glitches:
        ax.axvspan(g.start / fs * 1e3, max(g.end / fs * 1e3, g.start / fs * 1e3 + 0.15),
                   color=GLITCH_COLORS.get(g.kind, "k"), alpha=0.35, label=g.kind)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8, ncol=4)
    ax.set(xlabel="Time [ms]", ylabel="I [ADC counts]", title="Raw ADC (I channel) with detected glitch intervals")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_psd(iq, path: Path):
    fc = iq.meta.center_freq / 1e6
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(fc + iq.psd_freq / 1e6, iq.psd_db, lw=0.8, color="#1f4e79")
    ax.set(xlabel="Frequency [MHz]", ylabel="PSD [dB/Hz]", title="Welch power spectral density (whole capture, chunked)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_spectrogram(iq, path: Path):
    fc = iq.meta.center_freq / 1e6
    fig, ax = plt.subplots(figsize=(11, 4.2))
    vmin = np.percentile(iq.spec_db, 5)
    m = ax.pcolormesh(iq.spec_t * 1e3, fc + iq.spec_f / 1e6, iq.spec_db, shading="auto",
                      cmap="viridis", vmin=vmin, vmax=vmin + 60)
    for e in iq.emissions:
        if e["class"] in ("fhss_burst", "wideband_video_link"):
            c = "#ff4d4d" if e["class"] == "fhss_burst" else "#ffd700"
            ax.add_patch(plt.Rectangle((e["start_s"] * 1e3, fc + e["center_mhz"] - e["bandwidth_khz"] / 2e3),
                                       e["duration_ms"], e["bandwidth_khz"] / 1e3, fill=False, ec=c, lw=0.8))
    fig.colorbar(m, ax=ax, label="dB/Hz")
    ax.set(xlabel="Time [ms]", ylabel="Frequency [MHz]",
           title="Spectrogram  (red = FHSS control bursts, yellow = video downlink)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_error_timeline(err, path: Path):
    pm = err.per_minute
    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.bar(pm.index, pm["errors"], width=1 / 1440 * 0.9, color=np.where(pm["storm"], "#d62728", "#1f77b4"))
    ax.axhline(pm["threshold"].iat[0], ls="--", color="k", lw=0.8, label=f"storm threshold = {pm['threshold'].iat[0]:.0f}/min")
    ax.set(ylabel="ERROR events / min", title="Equipment error log: errors per minute (red = error storm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_fusion(fu, path: Path):
    fig = plt.figure(figsize=(11, 4.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_theta_zero_location("N"); ax1.set_theta_direction(-1)
    for cls, g in fu.radar.groupby("class"):
        ax1.scatter(np.deg2rad(g["azimuth_deg"]), g["range_m"], s=4 if cls == "radar_only" else 3,
                    c=CLASS_COLORS[cls], label=f"radar: {cls}", alpha=0.8)
    ax1.set_title("Radar plots (azimuth / range)", fontsize=10)
    ax1.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.25, -0.15))
    ax2 = fig.add_subplot(1, 2, 2)
    for cls, g in fu.rf.groupby("class"):
        ax2.scatter(g["timestamp"], g["bearing_deg"], s=6, c=CLASS_COLORS[cls], label=f"RF: {cls}")
    for _, e in fu.events.iterrows():
        ax2.axvspan(e["start"], e["end"], color="#d62728", alpha=0.08)
    ax2.set(ylabel="RF bearing [deg]", title="RF detections vs time (shaded = confirmed UAV)")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
def _img(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{path.stem}">'


def _table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<p class='muted'>None</p>"
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.6g}"
        return v
    return df.head(max_rows).map(fmt).to_html(index=False, border=0, classes="tbl")


CSS = """
:root{--bg:#f6f8fb;--card:#fff;--ink:#1d2433;--muted:#667085;--acc:#1f4e79;--bad:#d62728;--ok:#2e7d32}
body{margin:0;font-family:'Segoe UI','Malgun Gothic',sans-serif;background:var(--bg);color:var(--ink)}
header{background:var(--acc);color:#fff;padding:22px 32px}
header h1{margin:0;font-size:22px} header p{margin:4px 0 0;opacity:.85;font-size:13px}
main{max-width:1180px;margin:0 auto;padding:20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:var(--card);border-radius:10px;padding:14px 16px;box-shadow:0 1px 3px #0001}
.kpi b{display:block;font-size:24px;color:var(--acc)} .kpi span{font-size:12px;color:var(--muted)}
.kpi.bad b{color:var(--bad)}
section{background:var(--card);border-radius:10px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px #0001}
h2{font-size:17px;margin:0 0 10px;border-left:4px solid var(--acc);padding-left:8px}
img{max-width:100%;border-radius:6px}
.tbl{border-collapse:collapse;font-size:12.5px;width:100%} .tbl th{background:#eef2f7;text-align:left}
.tbl td,.tbl th{padding:5px 8px;border-bottom:1px solid #e5e9f0}
.muted{color:var(--muted);font-size:12px}
"""


def write_report(out_dir: str | Path, iq=None, err=None, fu=None, inputs: dict | None = None) -> Path:
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    figs, kpis, sections, summary = {}, [], [], {"generated": datetime.now().isoformat(timespec="seconds"),
                                                   "inputs": inputs or {}}

    if iq is not None:
        fs = iq.meta.sample_rate
        g = pd.DataFrame([x.as_row(fs) for x in iq.glitches])
        g.to_csv(out / "iq_glitches.csv", index=False)
        pd.DataFrame(iq.emissions).to_csv(out / "iq_emissions.csv", index=False)
        for name, fn in (("iq_overview", fig_iq_overview), ("psd", fig_psd), ("spectrogram", fig_spectrogram)):
            figs[name] = out / "figures" / f"{name}.png"; fn(iq, figs[name])
        kpis += [(f"{iq.n_samples / fs * 1e3:,.1f} ms", "IQ capture length"),
                 (len(iq.glitches), "ADC glitch intervals", "bad" if len(iq.glitches) else ""),
                 (len(iq.emitter_summary), "drone emitters identified", "bad" if iq.emitter_summary else "")]
        summary["iq"] = {"samples": iq.n_samples, "sample_rate": fs, "glitches": g.to_dict("records"),
                         "emitters": iq.emitter_summary}
        sections.append(("1. Raw ADC / IQ dump - glitch detection",
                         _img(figs["iq_overview"]) + _table(g)))
        sections.append(("2. FFT spectrum & drone emitter identification",
                         _img(figs["psd"]) + _img(figs["spectrogram"]) + _table(pd.DataFrame(iq.emitter_summary))))

    if fu is not None:
        fu.events.to_csv(out / "uav_events.csv", index=False)
        fu.rf.to_csv(out / "rf_fused.csv", index=False)
        figs["fusion"] = out / "figures" / "fusion.png"; fig_fusion(fu, figs["fusion"])
        kpis += [(fu.stats["confirmed_uav_tracks"], "confirmed UAV tracks (RF+radar)", "bad"),
                 (f"{fu.stats['radar_clutter_rejected_pct']}%", "radar clutter rejected")]
        summary["fusion"] = {"stats": fu.stats, "events": fu.events.to_dict("records")}
        sections.append(("3. RF / radar fusion - confirmed UAV events",
                         _img(figs["fusion"]) + _table(fu.events) +
                         "<p class='muted'>" + html.escape(json.dumps(fu.stats, default=_jsonable)) + "</p>"))

    if err is not None:
        err.per_minute.to_csv(out / "errors_per_minute.csv", index_label="minute")
        err.storms.to_csv(out / "error_storms.csv", index=False)
        figs["errors"] = out / "figures" / "error_timeline.png"; fig_error_timeline(err, figs["errors"])
        kpis += [(len(err.events), "log events parsed"),
                 (len(err.storms), "error storms", "bad" if len(err.storms) else ""),
                 (len(err.malformed_lines), "malformed lines skipped")]
        summary["errors"] = {"events": len(err.events), "storms": err.storms.to_dict("records"),
                             "top_codes": err.top_codes.to_dict("records"), "malformed": err.malformed_lines}
        sections.append(("4. Equipment error log - error storms",
                         _img(figs["errors"]) + _table(err.storms) + "<h3>Top codes</h3>" + _table(err.top_codes, 10)))

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable, ensure_ascii=False))

    kpi_html = "".join(f"<div class='kpi {k[2] if len(k) > 2 else ''}'><b>{k[0]}</b><span>{k[1]}</span></div>" for k in kpis)
    sec_html = "".join(f"<section><h2>{t}</h2>{body}</section>" for t, body in sections)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Anti-Drone Sensor Log Report</title>
<style>{CSS}</style></head><body>
<header><h1>Anti-Drone Sensor Log Analysis Report</h1>
<p>Generated {summary['generated']} by antidrone_analyzer - inputs: {html.escape(', '.join(map(str, (inputs or {}).values())))}</p></header>
<main><div class="kpis">{kpi_html}</div>{sec_html}</main></body></html>"""
    report = out / "report.html"
    report.write_text(page, encoding="utf-8")
    return report
