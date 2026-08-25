"""Aggregate mimicry-sweep results into tab_mimicry_stress.tex with F1-based Score.

Reads CSVs at results/proto/<dataset>/proto_<dataset>_<defense>_mimicry_eps<EEE>_r<RR>_s<SS>_metrics.csv
and aggregates by (dataset, defense) over all (epsilon, mr, seed) cells.
Reports F1, ASR, Score_F1 = F1 * (1-ASR) per cell, then averaged.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

VARIANTS = {
    "fedavg": "FedAvg", "krum": "Krum",
    "fedshield_v10_a025": "\\textbf{FedShield}",
    "fedshield_v10": "\\textbf{FedShield-Compact}",
}
DATASETS = {
    "mitbih": "MIT-BIH", "ciciomt": "CIC-IoMT-2024",
    "ptbxl": "PTB-XL", "physionet2017": "PhysioNet/CinC 2017",
}
RE = re.compile(
    r"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_mimicry_eps(?P<eps>\d{3})_r(?P<mr>\d{2})_s(?P<seed>\d{2})_metrics\.csv$"
)

def load() -> pd.DataFrame:
    rows = []
    for csv in Path("results/proto").rglob("proto_*_mimicry_*_metrics.csv"):
        m = RE.match(csv.name)
        if not m: continue
        if m.group("variant") not in VARIANTS or m.group("dataset") not in DATASETS:
            continue
        try: df = pd.read_csv(csv)
        except Exception: continue
        if len(df) == 0: continue
        last = df.iloc[-1]
        f1  = float(last.get("f1",  float("nan")))
        acc = float(last.get("acc", float("nan")))
        asr = float(last.get("asr", float("nan")))
        rows.append({
            "dataset": m.group("dataset"),
            "variant": m.group("variant"),
            "eps":     int(m.group("eps")) / 100.0,
            "mr":      int(m.group("mr")) / 100.0,
            "seed":    int(m.group("seed")),
            "acc": acc, "f1": f1, "asr": asr,
            "score": f1 * (1.0 - asr),
        })
    return pd.DataFrame(rows)

def main() -> None:
    df = load()
    print(f"Loaded {len(df)} mimicry runs.")
    g = df.groupby(["dataset", "variant"]).agg(
        f1=("f1", "mean"), asr=("asr", "mean"), score=("score", "mean")
    ).reset_index()

    winners = {}
    for ds in DATASETS:
        sub = g[g.dataset == ds].set_index("variant")
        for metric, lower in [("f1", False), ("asr", True), ("score", False)]:
            best = sub[metric].idxmin() if lower else sub[metric].idxmax()
            winners[(ds, metric)] = best

    def fmt(v, ds, metric, val):
        s = f"{val:.3f}"
        if winners[(ds, metric)] == v:
            return "\\cellcolor{gray!20}\\textbf{" + s + "}"
        return s

    rows_tex = []
    for v in VARIANTS:
        cells = []
        for ds in DATASETS:
            sub = g[(g.dataset == ds) & (g.variant == v)]
            if sub.empty: continue
            f1v = sub.iloc[0]["f1"]
            asrv = sub.iloc[0]["asr"]
            scv = sub.iloc[0]["score"]
            cells.append(fmt(v, ds, "f1", f1v))
            cells.append(fmt(v, ds, "asr", asrv))
            cells.append(fmt(v, ds, "score", scv))
        rows_tex.append(VARIANTS[v] + " & " + " & ".join(cells) + " \\\\")

    ds_hdr = " & ".join(["\\multicolumn{3}{c}{" + DATASETS[d] + "}" for d in DATASETS])
    cmid = " ".join([f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(DATASETS))])
    metric_hdr = " & ".join(["", *(["F1", "ASR", "Score"] * len(DATASETS))]) + " \\\\"

    tex = (
        "\\begin{table*}[!t]\n\\centering\\footnotesize\n"
        "\\caption{Krum-aware mimicry stress test. Malicious clients submit "
        "$\\Delta w_{\\mathrm{mal}} = \\mu_{\\mathrm{honest}} + "
        "\\varepsilon\\,\\lVert\\mu_{\\mathrm{honest}}\\rVert\\,v_{\\mathrm{attack}}$, "
        "with $v_{\\mathrm{attack}}$ a per-(client, parameter, round) reproducible "
        "unit-norm random direction. Four datasets, $\\varepsilon \\in \\{0.10, 0.25, 0.50\\}$, "
        "malicious-client ratios $\\rho_m \\in \\{0.2, 0.4\\}$, five seeds, 25 rounds: each "
        "cell is the mean of $30$ matched observations ($3$ $\\varepsilon$ $\\times$ $2$ "
        "$\\rho_m$ $\\times$ $5$ seeds). F1 is macro-averaged; lower ASR is better; "
        "higher F1 and Score $=\\mathrm{F1}\\cdot(1-\\mathrm{ASR})$ are better. "
        "Best result per column is highlighted.}\n"
        "\\label{tab:mimicry-stress}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{lrrrrrrrrrrrr}\n\\toprule\n"
        f"Method & {ds_hdr} \\\\\n{cmid}\n{metric_hdr}\n\\midrule\n"
        + "\n".join(rows_tex)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )
    out = Path("paper/tables/tab_mimicry_stress.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")

    print("\nPer-(dataset, method) summary (Score_F1):")
    print(g.pivot(index="variant", columns="dataset", values="score").round(3))

if __name__ == "__main__":
    main()
