"""Extract the relevant subset of the Kaggle PTB-XL zip into data/ptbxl/raw/.

The Kaggle download wraps everything under
``ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/``.
We strip that prefix and extract only:
    metadata CSVs (ptbxl_database.csv, scp_statements.csv)
    records100/**.hea + records100/**.dat (~550 MB at 100 Hz)
records500/ is skipped to save disk + IO.

Usage:
    python -m scripts.extract_ptbxl_zip --zip "data/PTB-XL ECG.zip"
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="./data/PTB-XL ECG.zip")
    ap.add_argument("--out_root", default="./data/ptbxl/raw")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    keep_patterns = (
        "/ptbxl_database.csv",
        "/scp_statements.csv",
        "/records100/",
    )

    with zipfile.ZipFile(args.zip, "r") as z:
        names = z.namelist()
        print(f"zip contains {len(names)} entries")
        # Find the version prefix (e.g. "ptb-xl-...-1.0.1/")
        roots = {n.split("/", 1)[0] for n in names if "/" in n}
        if len(roots) != 1:
            raise SystemExit(f"expected single top-level directory; found {roots}")
        prefix = roots.pop() + "/"
        print(f"prefix to strip: {prefix!r}")

        # Filter to keep set
        to_extract = [n for n in names
                      if n.startswith(prefix)
                      and any(p in n for p in keep_patterns)
                      and not n.endswith("/")]
        print(f"keeping {len(to_extract)} entries (records100 + metadata)")

        t0 = time.time()
        for i, name in enumerate(to_extract):
            rel = name[len(prefix):]
            dest = out_root / rel
            if dest.exists() and dest.stat().st_size > 0:
                continue  # cached
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / max(1.0, time.time() - t0)
                print(f"  extracted {i + 1}/{len(to_extract)}  ({rate:.0f}/s)", flush=True)

    print(f"done: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
