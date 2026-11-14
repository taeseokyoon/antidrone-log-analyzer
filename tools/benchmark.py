"""Reproduce the large-file benchmark in the README.

Builds a big IQ dump by repeating the bundled 2 MB sample N times (default
500 -> 1 GB), analyzes it with the normal chunked reader and checks that every
injected glitch was found exactly N times.

    python tools/benchmark.py              # 1 GB (500 copies)
    python tools/benchmark.py --copies 50  # 100 MB quick check

Peak memory (ru_maxrss) is reported on Linux/macOS; on Windows it is skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antidrone_analyzer.iq_analysis import analyze_iq  # noqa: E402


def peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copies", type=int, default=500, help="how many times to repeat samples/iq_dump.bin")
    ap.add_argument("--workdir", help="where to write the big file (default: temp folder, deleted afterwards)")
    a = ap.parse_args()

    src = ROOT / "samples" / "iq_dump.bin"
    tmp = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="adla_bench_"))
    tmp.mkdir(parents=True, exist_ok=True)
    big = tmp / "iq_dump.bin"
    chunk = src.read_bytes()
    with open(big, "wb") as fh:
        for _ in range(a.copies):
            fh.write(chunk)
    shutil.copy(src.with_suffix(".json"), big.with_suffix(".json"))
    size_mb = big.stat().st_size / 1e6
    print(f"[bench] built {big} ({size_mb:,.0f} MB, {a.copies} copies)")

    expected = Counter(g.kind for g in analyze_iq(src).glitches)
    t0 = time.time()
    res = analyze_iq(big)
    dt = time.time() - t0
    found = Counter(g.kind for g in res.glitches)
    want = {k: v * a.copies for k, v in expected.items()}

    print(f"[bench] samples       : {res.n_samples:,}")
    print(f"[bench] time          : {dt:,.1f} s  ({size_mb / dt:,.1f} MB/s)")
    rss = peak_rss_mb()
    print(f"[bench] peak memory   : {rss:,.0f} MB" if rss else "[bench] peak memory   : n/a on this OS")
    print(f"[bench] glitches found: {dict(found)}")
    print(f"[bench] expected      : {want}")
    ok = dict(found) == want
    print("[bench] RESULT        :", "PASS (no missing / no duplicate)" if ok else "FAIL")

    if not a.workdir:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
