"""Raw IQ / ADC dump analysis: glitch detection, PSD and emitter detection.

The dump is opened with ``numpy.memmap`` and processed in fixed-size chunks,
so files of several gigabytes can be scanned with constant memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage, signal


@dataclass
class IQMeta:
    sample_rate: float
    center_freq: float
    full_scale: int = 32767

    @classmethod
    def load(cls, bin_path: Path) -> "IQMeta":
        meta_path = Path(bin_path).with_suffix(".json")
        m = json.loads(meta_path.read_text())
        return cls(float(m["sample_rate"]), float(m["center_freq"]), int(m.get("full_scale", 32767)))


@dataclass
class Interval:
    kind: str
    start: int          # complex-sample index (inclusive)
    end: int            # complex-sample index (exclusive)
    peak: float = 0.0

    def as_row(self, fs: float) -> dict:
        return {"type": self.kind, "start_s": self.start / fs, "end_s": self.end / fs,
                "duration_us": (self.end - self.start) / fs * 1e6, "peak": round(self.peak, 1)}


@dataclass
class IQResult:
    meta: IQMeta
    n_samples: int
    bin_path: Path | None = None
    glitches: list = field(default_factory=list)
    psd_freq: np.ndarray | None = None
    psd_db: np.ndarray | None = None
    spec_t: np.ndarray | None = None
    spec_f: np.ndarray | None = None
    spec_db: np.ndarray | None = None
    emissions: list = field(default_factory=list)
    emitter_summary: list = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _merge(intervals: list[Interval], gap: int) -> list[Interval]:
    """Merge same-kind intervals closer than ``gap`` samples (also across chunk edges)."""
    out: list[Interval] = []
    for iv in sorted(intervals, key=lambda i: (i.kind, i.start)):
        if out and out[-1].kind == iv.kind and iv.start - out[-1].end <= gap:
            out[-1].end = max(out[-1].end, iv.end)
            out[-1].peak = max(out[-1].peak, iv.peak)
        else:
            out.append(Interval(iv.kind, iv.start, iv.end, iv.peak))
    return sorted(out, key=lambda i: i.start)


# --------------------------------------------------------------------------
# glitch detection
# --------------------------------------------------------------------------
def detect_glitches(raw: np.ndarray, offset: int, full_scale: int,
                    dc_block: int = 4096, dc_sigma: float = 8.0,
                    dc_noise_std: float | None = None) -> list[Interval]:
    """Find saturation, impulse spikes, dropouts and DC steps in one chunk.

    ``raw`` is the interleaved int16 array of this chunk, ``offset`` the
    complex-sample index of its first sample.
    """
    i = raw[0::2].astype(np.int32)
    q = raw[1::2].astype(np.int32)
    peak = np.maximum(np.abs(i), np.abs(q))
    found: list[Interval] = []

    # (1) high-amplitude events -> spike (short) or saturation (long / clipped)
    # Inside an overload the envelope dips below threshold for a few hundred
    # samples, so close gaps up to 256 samples before measuring the run.
    hot = peak > 0.7 * full_scale
    hot = ndimage.binary_closing(hot, structure=np.ones(256)) | hot
    for s, e in _runs(hot):
        clipped = np.count_nonzero(peak[s:e] >= full_scale - 1)
        kind = "spike" if (e - s) <= 32 and clipped < 4 else "saturation"
        found.append(Interval(kind, offset + s, offset + e, float(peak[s:e].max())))

    # (2) dropouts: I and Q both exactly zero for >= 16 samples
    zero = (i == 0) & (q == 0)
    for s, e in _runs(zero):
        if e - s >= 16:
            found.append(Interval("dropout", offset + s, offset + e, 0.0))

    # (3) DC step: block mean far from zero relative to the noise level
    nblk = len(i) // dc_block
    if nblk >= 1:
        mi = i[: nblk * dc_block].reshape(nblk, dc_block).mean(axis=1)
        mq = q[: nblk * dc_block].reshape(nblk, dc_block).mean(axis=1)
        noise = dc_noise_std if dc_noise_std else np.median(np.abs(i)) * 1.4826
        sigma_mean = max(noise / np.sqrt(dc_block), 1.0)
        bad = np.hypot(mi, mq) > dc_sigma * sigma_mean
        # A block that contains an overload / spike / dropout already has a
        # biased mean (clipping is not symmetric), so it must not be reported
        # a second time as a DC step.
        explained = (hot | zero)[: nblk * dc_block].reshape(nblk, dc_block).any(axis=1)
        bad &= ~explained
        for s, e in _runs(bad):
            found.append(Interval("dc_step", offset + s * dc_block, offset + e * dc_block,
                                  float(np.hypot(mi[s:e], mq[s:e]).max())))
    return found


# --------------------------------------------------------------------------
# spectrum / emitter detection
# --------------------------------------------------------------------------
def detect_emissions(f: np.ndarray, t: np.ndarray, sxx_db: np.ndarray,
                     threshold_db: float = 12.0) -> list[dict]:
    """Label connected time-frequency regions above the noise floor."""
    floor = np.median(sxx_db)
    mask = sxx_db > floor + threshold_db          # shape (freq, time)
    df = f[1] - f[0]
    dt = t[1] - t[0] if len(t) > 1 else 0.0
    out = []

    # Broadband frames (spikes / ADC overload light up the whole band). They
    # would bridge every emitter into one huge region, so cut them out first.
    broad = mask.mean(axis=0) > 0.5
    for s, e in _runs(broad):
        out.append({"start_s": float(t[s] - dt / 2), "duration_ms": float((e - s) * dt * 1e3),
                    "center_mhz": 0.0, "bandwidth_khz": float((f[-1] - f[0] + df) / 1e3),
                    "peak_db": float(sxx_db[:, s:e].max())})
    mask[:, broad] = False

    # Join leakage fragments of the same tone (few bins apart in frequency),
    # then drop isolated single-cell noise hits.
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 1)))
    labels, _ = ndimage.label(mask)
    lin = 10 ** (sxx_db / 10)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        fs_, ts_ = sl
        if (fs_.stop - fs_.start) * (ts_.stop - ts_.start) < 6:
            continue
        t0, t1 = t[ts_.start] - dt / 2, t[ts_.stop - 1] + dt / 2
        # Bandwidth from the time-averaged spectrum of this component at -10 dB.
        # The bounding box overstates it: burst on/off edges splatter energy
        # across neighbouring bins for one frame.
        own = labels[sl] == k
        prof = np.where(own, lin[sl], 0).sum(axis=1) / np.maximum(own.sum(axis=1), 1)
        prof_db = 10 * np.log10(prof + 1e-20)
        keep = np.flatnonzero(prof_db > prof_db.max() - 10)
        fb = f[fs_.start + keep]
        f0, f1 = fb.min(), fb.max() + df
        out.append({"start_s": float(t0), "duration_ms": float((t1 - t0) * 1e3),
                    "center_mhz": float((f0 + f1) / 2 / 1e6), "bandwidth_khz": float((f1 - f0) / 1e3),
                    "peak_db": float(sxx_db[sl].max())})
    for e in out:
        e["class"] = classify_emission(e)
    return sorted(out, key=lambda e: e["start_s"])


def classify_emission(e: dict) -> str:
    bw, dur = e["bandwidth_khz"], e["duration_ms"]
    if bw > 3000 and dur < 0.5:
        return "broadband_transient"
    if abs(e["center_mhz"]) < 0.08 and bw < 150:
        return "dc_offset"
    if bw >= 800 and dur >= 2.0:
        return "wideband_video_link"
    if bw < 800 and 0.2 <= dur <= 1.5:
        return "fhss_burst"
    return "unknown"


def summarize_emitters(emissions: list[dict]) -> list[dict]:
    """Turn individual emissions into emitter-level findings."""
    out = []
    fh = [e for e in emissions if e["class"] == "fhss_burst"]
    if len(fh) >= 5:
        starts = np.array([e["start_s"] for e in fh])
        chans = sorted({round(e["center_mhz"] * 2) / 2 for e in fh})
        out.append({
            "emitter": "FHSS control link (drone C2 uplink)",
            "bursts": len(fh),
            "hop_channels": len(chans),
            "channels_mhz": ", ".join(f"{c:+.1f}" for c in chans),
            "burst_ms": round(float(np.median([e["duration_ms"] for e in fh])), 2),
            "period_ms": round(float(np.median(np.diff(starts))) * 1e3, 2),
        })
    vid = [e for e in emissions if e["class"] == "wideband_video_link"]
    if vid:
        total = sum(e["duration_ms"] for e in vid)
        main = max(vid, key=lambda e: e["duration_ms"])
        out.append({
            "emitter": "Wideband video downlink",
            "segments": len(vid),
            "center_mhz": round(main["center_mhz"], 2),
            "bandwidth_khz": round(main["bandwidth_khz"], 0),
            "first_seen_s": round(min(e["start_s"] for e in vid), 4),
            "on_time_ms": round(total, 1),
        })
    return out


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------
def read_chunk(path: Path, start: int, count: int) -> np.ndarray:
    """Read ``count`` complex samples starting at ``start`` as interleaved int16.

    Explicit reads (instead of one big ``np.memmap``) keep resident memory at
    one chunk: with memmap every page touched stays counted in RSS, which hit
    ~1.1 GB on a 1 GB dump.
    """
    return np.fromfile(path, dtype="<i2", count=2 * count, offset=4 * start)


def analyze_iq(bin_path: str | Path, chunk_samples: int = 1 << 20,
               nfft: int = 1024, spec_seconds: float = 0.2) -> IQResult:
    bin_path = Path(bin_path)
    meta = IQMeta.load(bin_path)
    fs = meta.sample_rate
    n = bin_path.stat().st_size // 4          # 2 x int16 per complex sample
    res = IQResult(meta=meta, n_samples=n, bin_path=bin_path)

    # global noise estimate from the first chunk (robust, ignores bursts)
    first = read_chunk(bin_path, 0, min(n, chunk_samples)).astype(np.float32)
    noise_std = float(np.median(np.abs(first[0::2])) * 1.4826)

    glitches: list[Interval] = []
    psd_acc, psd_w = None, 0
    for c0 in range(0, n, chunk_samples):
        c1 = min(n, c0 + chunk_samples)
        chunk = read_chunk(bin_path, c0, c1 - c0)
        glitches += detect_glitches(chunk, c0, meta.full_scale, dc_noise_std=noise_std)
        x = chunk[0::2].astype(np.float32) + 1j * chunk[1::2].astype(np.float32)
        if len(x) >= nfft:
            fr, p = signal.welch(x, fs=fs, nperseg=nfft, return_onesided=False,
                                 scaling="density", detrend=False)
            psd_acc = p * len(x) if psd_acc is None else psd_acc + p * len(x)
            psd_w += len(x)
    res.glitches = _merge(glitches, gap=64)

    if psd_acc is not None:
        order = np.argsort(fr)
        res.psd_freq = fr[order]
        res.psd_db = 10 * np.log10(psd_acc[order] / psd_w + 1e-20)

    # spectrogram of the first `spec_seconds` (enough for a report figure)
    m = min(n, int(spec_seconds * fs))
    seg = read_chunk(bin_path, 0, m).astype(np.float32)
    x = seg[0::2] + 1j * seg[1::2]
    # Blackman-Harris keeps side-lobes ~90 dB down, so a strong hop tone does
    # not leak into neighbouring bins and get counted as several emitters.
    f, t, sxx = signal.spectrogram(x, fs=fs, window="blackmanharris", nperseg=256, noverlap=0,
                                   return_onesided=False, detrend=False, mode="psd")
    order = np.argsort(f)
    res.spec_f, res.spec_t = f[order], t
    res.spec_db = 10 * np.log10(sxx[order] + 1e-20)
    res.emissions = detect_emissions(res.spec_f, res.spec_t, res.spec_db)
    res.emitter_summary = summarize_emitters(res.emissions)
    return res
