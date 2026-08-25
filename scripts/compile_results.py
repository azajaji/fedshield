"""Parse all completed proto_*_metrics.csv files and produce paper-ready tables.

Bypasses the broken print path in proto_bench (which died on a Unicode
delta character in cp1252 stdout). Reads the cached CSVs directly,
groups by (dataset, variant, attack, mr, seed[, num_clients]), computes
mean/std across seeds, runs paired t-tests vs FedShield, and prints
one table per sweep.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

try:
    from scipy import stats as _stats
except ImportError:
    _stats = None


# Known attack tokens — used to anchor the regex so variant names containing
# underscores (e.g., "A1_no_edge_ae", "v10_ens90") don't get truncated.
KNOWN_ATTACKS = [
    "label_flip+backdoor+sign_flip+scaling",
    "sign_flip+scaling",
    "backdoor+scaling",
    "sign_flip",
    "label_flip",
    "backdoor",
    "scaling",
    "noise_update",
    "clean",
]
_ATTACK_ALT = "|".join(re.escape(a) for a in KNOWN_ATTACKS)

# Filename: proto_<dataset>_<variant>_<attack>_r<mr>_s<seed>[_n<N>]_metrics.csv
FILENAME_RE = re.compile(
    rf"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    rf"(?P<attack>{_ATTACK_ALT})_r(?P<mr>\d{{2}})_s(?P<seed>\d{{2}})"
    rf"(?:_n(?P<n>\d{{2}}))?_metrics\.csv$"
)


def parse_all(root: Path = Path("./results/proto")) -> pd.DataFrame:
    rows = []
    for csv in root.rglob("proto_*_metrics.csv"):
        m = FILENAME_RE.match(csv.name)
        if not m:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0:
            continue
        last = df.iloc[-1]
        rows.append({
            "dataset":   m.group("dataset"),
            "variant":   m.group("variant"),
            "attack":    m.group("attack"),
            "mr":        float(m.group("mr")) / 100,
            "seed":      int(m.group("seed")),
            "num_clients": int(m.group("n")) if m.group("n") else None,
            "acc":       float(last.get("acc", float("nan"))),
            "asr":       float(last.get("asr", float("nan"))),
            "score":     float(last.get("defense_score", float("nan"))),
            "lat_edge":  float(last.get("latency_edge_ms", float("nan"))),
            "lat_srv":   float(last.get("latency_server_ms", float("nan"))),
            "comm_kb":   float(last.get("comm_bytes_per_client", float("nan"))) / 1024.0,
            "ram_mb":    float(last.get("edge_ram_mb", float("nan"))),
        })
    return pd.DataFrame(rows)


def per_cell_means(df: pd.DataFrame, group_keys=("dataset", "variant", "attack", "mr")) -> pd.DataFrame:
    return (df.groupby(list(group_keys))
              .agg(mean_score=("score", "mean"),
                   std_score=("score", "std"),
                   n_seeds=("seed", "nunique"))
              .reset_index())


def headline_table(df: pd.DataFrame, dataset: str) -> None:
    headline_variants = ("fedavg", "krum", "multi_krum", "trimmed_mean",
                         "foolsgold", "fltrust", "median",
                         "fedshield_v10", "fedshield_v10_a025")
    sub = df[(df.dataset == dataset)
             & (df.variant.isin(headline_variants))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        print(f"  no headline data for {dataset}")
        return

    print(f"\n========== HEADLINE MATRIX — {dataset.upper()} ==========")
    pivot = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .unstack(["attack", "mr"])
                .reindex(index=list(headline_variants))
                .dropna(how="all"))
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"))

    # Aggregate leaderboard
    cell_means = sub.groupby(["variant", "attack", "mr"])["score"].mean().reset_index()
    leaderboard = (cell_means.groupby("variant")["score"].agg(["mean", "sum", "count"])
                   .reset_index()
                   .sort_values("mean", ascending=False))
    print(f"\n--- Leaderboard ({dataset}) ---")
    for _, row in leaderboard.iterrows():
        marker = " <- ours" if row["variant"] == "fedshield_v10" else ""
        print(f"  {row['variant']:<18} mean={row['mean']:.3f}  sum={row['sum']:.3f}  cells={int(row['count'])}{marker}")

    # Paired t-test FedShield vs each baseline
    if _stats is not None and "fedshield_v10" in sub.variant.values:
        print(f"\n--- t-test (fedshield_v10 vs each baseline, paired by (attack, mr, seed)) ---")
        pivot_seeds = (sub.pivot_table(index=["attack", "mr", "seed"], columns="variant",
                                       values="score", aggfunc="first"))
        if "fedshield_v10" in pivot_seeds.columns:
            print(f"  {'baseline':<18} {'mean_diff':>10} {'t':>8} {'p':>10} {'n':>5}")
            for v in headline_variants:
                if v == "fedshield_v10" or v not in pivot_seeds.columns:
                    continue
                pair = pivot_seeds[["fedshield_v10", v]].dropna()
                if len(pair) < 3:
                    continue
                t, p = _stats.ttest_rel(pair["fedshield_v10"], pair[v])
                d = float((pair["fedshield_v10"] - pair[v]).mean())
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {v:<18} {d:+10.3f} {float(t):+8.2f} {float(p):10.4f} {len(pair):>5}  {sig}")


def ablation_table(df: pd.DataFrame) -> None:
    abl_variants = ("fedshield_v10", "A1_no_edge_ae", "A2_no_norm_clip",
                    "A3_mean_ref", "A4a_mean_only", "A4b_median_only", "A5_buggy_f")
    sub = df[(df.dataset == "mitbih")
             & (df.variant.isin(abl_variants))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return
    print(f"\n========== ABLATION (MIT-BIH) ==========")
    pivot = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .unstack(["attack", "mr"]).reindex(index=list(abl_variants)))
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"))
    print(f"\n--- Ablation leaderboard ---")
    means = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean().sort_values(ascending=False))
    for v, s in means.items():
        delta = s - means.get("fedshield_v10", 0.0)
        print(f"  {v:<22} mean={s:.3f}  delta_vs_full={delta:+.3f}")


def cohort_table(df: pd.DataFrame) -> None:
    sub = df[(df.dataset == "mitbih") & (df.num_clients.notna())].copy()
    if len(sub) == 0:
        print("\n  (no cohort data)")
        return
    print(f"\n========== COHORT-SIZE SENSITIVITY (MIT-BIH) ==========")
    means = (sub.groupby(["num_clients", "variant", "attack", "mr"])["score"].mean()
                .reset_index().groupby(["num_clients", "variant"])["score"].mean()
                .reset_index().pivot(index="num_clients", columns="variant", values="score"))
    print(means.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"))


def rho_curve(df: pd.DataFrame) -> None:
    # Filter to sign_flip cells only (the rho-sweep panel)
    sub = df[(df.dataset == "mitbih")
             & (df.attack == "sign_flip")
             & (df.variant.isin(["krum", "fedshield_v10"]))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return
    print(f"\n========== RHO-SWEEP CURVE — sign_flip on MIT-BIH ==========")
    means = sub.groupby(["mr", "variant"])["score"].mean().unstack("variant")
    print(means.to_string(float_format=lambda x: f"{x:.3f}", na_rep="—"))


def cost_table(df: pd.DataFrame) -> None:
    headline = ("fedavg", "krum", "multi_krum", "trimmed_mean",
                "foolsgold", "fltrust", "median", "fedshield_v10")
    sub = df[df.variant.isin(headline)].copy()
    print(f"\n========== COST SUMMARY (per-defense averages, all datasets) ==========")
    cost = sub.groupby("variant")[["lat_edge", "lat_srv", "comm_kb", "ram_mb"]].mean().reset_index()
    cost = cost.sort_values("variant")
    print(cost.to_string(index=False, float_format=lambda x: f"{x:.2f}", na_rep="—"))


def sensitivity_table(df: pd.DataFrame, dataset: str = "mitbih") -> None:
    sens_variants = [v for v in df.variant.unique()
                     if v.startswith("S_") or v == "fedshield_v10"]
    sub = df[(df.dataset == dataset) & (df.variant.isin(sens_variants))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return
    print(f"\n========== SENSITIVITY ({dataset.upper()}) ==========")
    means = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean()
                .sort_values(ascending=False))
    print("--- mean score by variant (alpha + (m2,mm) sweep) ---")
    for v, s in means.items():
        marker = " <- chosen" if v == "fedshield_v10" else ""
        print(f"  {v:<14} mean={s:.3f}{marker}")


def tuned_baselines_table(df: pd.DataFrame) -> None:
    tuned = ("krum_f1", "krum_f2", "krum_f3",
             "multi_krum_m3", "multi_krum_m5", "multi_krum_m7",
             "trimmed_b10", "trimmed_b20", "trimmed_b30",
             "fltrust_r100", "fltrust_r200", "fltrust_r400",
             "fedshield_v10")
    sub = df[(df.dataset == "mitbih") & (df.variant.isin(tuned))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return
    print(f"\n========== TUNED BASELINES (MIT-BIH, fairness check) ==========")
    means = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean()
                .sort_values(ascending=False))
    # Best-of each baseline family
    families = {"krum": ["krum_f1", "krum_f2", "krum_f3"],
                "multi_krum": ["multi_krum_m3", "multi_krum_m5", "multi_krum_m7"],
                "trimmed_mean": ["trimmed_b10", "trimmed_b20", "trimmed_b30"],
                "fltrust": ["fltrust_r100", "fltrust_r200", "fltrust_r400"]}
    print("--- per-variant ---")
    for v, s in means.items():
        marker = " <- ours" if v == "fedshield_v10" else ""
        print(f"  {v:<18} mean={s:.3f}{marker}")
    print("\n--- best-of each family vs FedShield ---")
    fs_score = means.get("fedshield_v10", float("nan"))
    print(f"  fedshield_v10      = {fs_score:.3f}")
    for fam, vs in families.items():
        best_v = max(vs, key=lambda x: means.get(x, -1))
        best_s = means.get(best_v, float("nan"))
        delta = fs_score - best_s
        print(f"  best-tuned {fam:<13} = {best_s:.3f}  ({best_v})   delta_to_FedShield={delta:+.3f}")


def design_rationale_table(df: pd.DataFrame) -> None:
    rationale = ("fedshield_v10", "single_mean_top2", "single_median_top7",
                 "single_trim_top7", "ens_mean+trim", "ens_median+trim",
                 "ens_mean+geomed", "bulyan_classic", "ens_no_krum_filter",
                 "ens_max", "ens_min", "ens_alarm_gated")
    sub = df[(df.dataset == "mitbih") & (df.variant.isin(rationale))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return
    print(f"\n========== DESIGN-RATIONALE (MIT-BIH, 17 cells x 5 seeds) ==========")
    means = (sub.groupby(["variant", "attack", "mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean()
                .reset_index().rename(columns={"score": "mean"}))
    fs_score = float(means.loc[means.variant == "fedshield_v10", "mean"].iloc[0]) if "fedshield_v10" in means.variant.values else float("nan")
    means["delta_vs_v10"] = means["mean"] - fs_score
    means = means.sort_values("mean", ascending=False)
    print(f"  {'variant':<22} {'mean':>8} {'delta':>9}  {'verdict':<30}")
    for _, row in means.iterrows():
        verdict = ""
        v = row["variant"]
        if v == "fedshield_v10":            verdict = "<- chosen design"
        elif v == "single_mean_top2":       verdict = "ensemble > single mean"
        elif v == "single_median_top7":     verdict = "ensemble > single median"
        elif v == "single_trim_top7":       verdict = "ensemble > single trim"
        elif v.startswith("ens_mean+"):     verdict = "alt component pair"
        elif v == "ens_median+trim":        verdict = "alt component pair"
        elif v == "bulyan_classic":         verdict = "parallel > sequential"
        elif v == "ens_no_krum_filter":     verdict = "Krum prefilter is needed"
        elif v == "ens_max":                verdict = "convex > max"
        elif v == "ens_min":                verdict = "convex > min"
        elif v == "ens_alarm_gated":        verdict = "convex > alarm-gated"
        print(f"  {v:<22} {row['mean']:>8.3f} {row['delta_vs_v10']:>+9.3f}  {verdict}")


def main() -> None:
    df = parse_all()
    print(f"Parsed {len(df)} proto runs across {df.dataset.nunique()} datasets, "
          f"{df.variant.nunique()} variants, {df.attack.nunique()} attack types.")
    # iterate all datasets; missing ones print "no data" and skip
    for d in ("mitbih", "ptbxl", "physionet2017", "physionet2020", "ciciomt", "wesad"):
        headline_table(df, d)
    ablation_table(df)
    design_rationale_table(df)
    sensitivity_table(df, "mitbih")
    sensitivity_table(df, "ptbxl")
    tuned_baselines_table(df)
    rho_curve(df)
    cohort_table(df)
    cost_table(df)


if __name__ == "__main__":
    main()
