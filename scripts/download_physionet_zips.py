"""Download PhysioNet 2017 + 2020 challenge ZIPs.

Both are publicly reachable with no auth. Resumable: existing files skip.

Usage:
    python -m scripts.download_physionet_zips
    python -m scripts.download_physionet_zips --skip-2020  # smaller download
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

PHYSIONET_2017_ZIP = (
    "https://physionet.org/static/published-projects/challenge-2017/"
    "af-classification-from-a-short-single-lead-ecg-recording-the-"
    "physionetcomputing-in-cardiology-challenge-2017-1.0.0.zip"
)
PHYSIONET_2020_ZIP = (
    "https://physionet.org/static/published-projects/challenge-2020/"
    "classification-of-12-lead-ecgs-the-physionetcomputing-in-"
    "cardiology-challenge-2020-1.0.2.zip"
)


def fetch(url: str, dest: Path, chunk: int = 1 << 22) -> None:
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"[cached] {dest.name}: {dest.stat().st_size / 1024 / 1024:.0f} MB")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url} -> {dest.name}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f:
            done = 0
            t0 = time.time()
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total and done % (chunk * 8) == 0:
                    pct = 100.0 * done / total
                    rate = done / max(1.0, time.time() - t0) / 1024 / 1024
                    print(f"\r  {pct:5.1f}%  {rate:6.1f} MB/s", end="", flush=True)
    tmp.rename(dest)
    print(f"\r  {dest.name}: done ({dest.stat().st_size / 1024 / 1024:.0f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="./data")
    ap.add_argument("--skip-2017", action="store_true")
    ap.add_argument("--skip-2020", action="store_true")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    if not args.skip_2017:
        fetch(PHYSIONET_2017_ZIP, out_root / "physionet_2017.zip")
    if not args.skip_2020:
        fetch(PHYSIONET_2020_ZIP, out_root / "physionet_2020.zip")
    print("Done.")


if __name__ == "__main__":
    main()
