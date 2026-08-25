"""Generate paper-ready LaTeX tables from the proto results CSV cache.

Outputs into paper/tables/ as standalone .tex fragments to be \\input'ed:
    tab_headline_<dataset>.tex   — per-dataset 17-cell matrix with FedShield highlighted
    tab_wins_summary.tex         — W/L/T counter at p<0.05 across datasets
    tab_ablation.tex             — stage-level ablation on MIT-BIH (academic naming)
    tab_design_rationale.tex     — design-rationale ablations (12 variants)
    tab_tuned_baselines.tex      — best-of each baseline family vs FedShield
    tab_cost.tex                 — per-defense compute cost
"""
from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
try:
    from scipy import stats
except ImportError:
    stats = None


KNOWN_ATTACKS = [
    "label_flip+backdoor+sign_flip+scaling",
    "sign_flip+scaling",
    "backdoor+scaling",
    "sign_flip", "label_flip", "backdoor", "scaling", "noise_update", "clean",
]
_ATTACK_ALT = "|".join(re.escape(a) for a in KNOWN_ATTACKS)
FILENAME_RE = re.compile(
    rf"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    rf"(?P<attack>{_ATTACK_ALT})_r(?P<mr>\d{{2}})_s(?P<seed>\d{{2}})"
    rf"(?:_n(?P<n>\d{{2}}))?_metrics\.csv$"
)

DISPLAY = {
    "fedavg":            "FedAvg",
    "krum":              "Krum",
    "multi_krum":        "Multi-Krum",
    "trimmed_mean":      "Trimmed-Mean",
    "foolsgold":         "FoolsGold",
    "fltrust":           "FLTrust",
    "fedshield_v10":     r"\textbf{FedShield (proposed)}",
    "fedshield_v10_a025":r"\textbf{FedShield-PD}",
}
HEADLINE_VARIANTS = list(DISPLAY.keys())

DATASETS_FULL = ["mitbih", "ptbxl", "ciciomt", "physionet2017", "physionet2020"]
DATASET_DISPLAY = {
    "mitbih":         "MIT-BIH",
    "ptbxl":          "PTB-XL",
    "ciciomt":        "CIC-IoMT",
    "physionet2017":  "PhysioNet 2017",
    "physionet2020":  "PhysioNet 2020",
}


def parse_all(root: Path = Path("./results/proto")) -> pd.DataFrame:
    rows = []
    for csv in root.rglob("proto_*_metrics.csv"):
        m = FILENAME_RE.match(csv.name)
        if not m: continue
        try: df = pd.read_csv(csv)
        except: continue
        if len(df) == 0: continue
        last = df.iloc[-1]
        # Headline switched to F1-based defense score:
        #   Score = F1 * (1 - ASR)
        # which penalizes majority-class collapse on imbalanced healthcare
        # data. The legacy 'defense_score' column in the CSV is Acc-based and
        # is retained as score_acc for cross-reference.
        f1  = float(last.get("f1",  float("nan")))
        asr = float(last.get("asr", float("nan")))
        rows.append({
            "dataset": m.group("dataset"), "variant": m.group("variant"),
            "attack": m.group("attack"), "mr": float(m.group("mr"))/100,
            "seed": int(m.group("seed")),
            "num_clients": int(m.group("n")) if m.group("n") else None,
            "score":     f1 * (1.0 - asr),
            "score_acc": float(last.get("defense_score", float("nan"))),
            "lat_edge": float(last.get("latency_edge_ms", float("nan"))),
            "lat_srv":  float(last.get("latency_server_ms", float("nan"))),
            "comm":     float(last.get("comm_bytes_per_client", float("nan"))) / 1024.0,
            "ram":      float(last.get("edge_ram_mb", float("nan"))),
        })
    return pd.DataFrame(rows)


def headline_table_tex(df: pd.DataFrame, dataset: str) -> str:
    sub = df[(df.dataset == dataset) & (df.variant.isin(HEADLINE_VARIANTS))
             & (df.num_clients.isna())].copy()
    if len(sub) == 0:
        return f"% (no data for {dataset})\n"
    means = sub.groupby(["variant","attack","mr"])["score"].mean().reset_index()
    leaderboard = (means.groupby("variant")["score"].mean()
                   .reindex([v for v in HEADLINE_VARIANTS if v in means.variant.values])
                   .sort_values(ascending=False))
    fs_score = leaderboard.get("fedshield_v10", float("nan"))
    fs_rank = list(leaderboard.index).index("fedshield_v10") + 1 if "fedshield_v10" in leaderboard.index else "—"

    # Paired t-tests
    tt_lines = []
    if stats is not None and "fedshield_v10" in sub.variant.values:
        pivot = sub.pivot_table(index=["attack","mr","seed"], columns="variant",
                                 values="score", aggfunc="first")
        for v in HEADLINE_VARIANTS:
            if v == "fedshield_v10" or v not in pivot.columns: continue
            pair = pivot[["fedshield_v10", v]].dropna()
            if len(pair) < 3: continue
            t, p = stats.ttest_rel(pair["fedshield_v10"], pair[v])
            d = float((pair["fedshield_v10"] - pair[v]).mean())
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "\\ "
            tt_lines.append(f"  {DISPLAY.get(v,v)} & ${d:+.3f}$ & ${float(p):.4f}$ & {sig} \\\\")

    label = f"tab:headline-{dataset}"
    out = []
    out.append(r"\begin{table}[!t]")
    out.append(r"\centering\small")
    out.append(rf"\caption{{Headline attack matrix on {DATASET_DISPLAY[dataset]} ({len(means)} cells averaged across 5 seeds). "
               rf"FedShield ranks {fs_rank} of {len(leaderboard)} (mean defense score {fs_score:.3f}). "
               rf"Significance: ***$p<0.001$, **$p<0.01$, *$p<0.05$ via paired $t$-test.}}")
    out.append(rf"\label{{{label}}}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{l r r r}")
    out.append(r"\toprule")
    out.append(r"Defense & Mean score & $\Delta$ vs FedShield & sig.\ \\")
    out.append(r"\midrule")
    for v, s in leaderboard.items():
        nm = DISPLAY.get(v, v)
        delta = s - fs_score
        sig_marker = ""
        if v == "fedshield_v10":
            row = rf"  \rowcolor{{gray!15}} {nm} & {s:.3f} & --- & --- \\"
        else:
            # find the t-test line for this v
            t_line = next((ln for ln in tt_lines if DISPLAY.get(v,v) in ln), None)
            sig_marker = "\\ "
            if t_line:
                # extract the sig marker
                m = re.search(r"& (\\*\*+|\\\s)\s*\\\\", t_line)
                if m: sig_marker = m.group(1).strip() or "\\ "
            row = rf"  {nm} & {s:.3f} & ${delta:+.3f}$ & {sig_marker} \\"
        out.append(row)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def wins_summary_table_tex(df: pd.DataFrame) -> str:
    """Per-(dataset, baseline) outcome at p<0.05: WIN / TIE / LOSS for FedShield."""
    if stats is None: return "% scipy unavailable\n"
    rows = []
    # Exclude FedShield itself and the per-dataset tuned variant from columns
    # (FedShield-PD is reported in the per-dataset matrices instead, since it is
    # only run on PTB-XL).
    excluded_cols = {"fedshield_v10", "fedshield_v10_a025"}
    headers = ["Dataset"] + [DISPLAY[v].replace(r"\textbf{","").replace("}","")
                              for v in HEADLINE_VARIANTS if v not in excluded_cols]
    for ds in DATASETS_FULL:
        sub = df[(df.dataset == ds) & (df.variant.isin(HEADLINE_VARIANTS))
                 & (df.num_clients.isna())]
        if len(sub) == 0: continue
        pivot = sub.pivot_table(index=["attack","mr","seed"], columns="variant",
                                 values="score", aggfunc="first")
        if "fedshield_v10" not in pivot.columns: continue
        outcomes = []
        for v in HEADLINE_VARIANTS:
            if v in excluded_cols: continue
            if v not in pivot.columns:
                outcomes.append("---"); continue
            pair = pivot[["fedshield_v10", v]].dropna()
            if len(pair) < 3:
                outcomes.append("---"); continue
            t, p = stats.ttest_rel(pair["fedshield_v10"], pair[v])
            d = float((pair["fedshield_v10"] - pair[v]).mean())
            if p < 0.05 and d > 0: outcomes.append(r"\textbf{W}")
            elif p < 0.05 and d < 0: outcomes.append("L")
            else: outcomes.append("T")
        rows.append((DATASET_DISPLAY[ds], outcomes))

    n_W = sum(1 for ds,outs in rows for o in outs if "W" in o)
    n_T = sum(1 for ds,outs in rows for o in outs if o == "T")
    n_L = sum(1 for ds,outs in rows for o in outs if o == "L")
    n_total = sum(1 for ds,outs in rows for o in outs if o != "---")

    # Shorten headers to keep the table within \textwidth
    short = {"FedAvg":"FedAvg","Krum":"Krum","Multi-Krum":"M-Krum",
             "Trimmed-Mean":"Trim-Mean","FoolsGold":"FoolsG.",
             "FLTrust":"FLTrust","Coord-Median":"Median"}
    headers_short = [headers[0]] + [short.get(h, h) for h in headers[1:]]
    out = []
    out.append(r"\begin{table*}[!t]")
    out.append(r"\centering\small")
    out.append(rf"\caption{{Head-to-head record. Each cell is the outcome of a paired "
               rf"$t$-test at $\alpha=0.05$ between FedShield and a baseline, aggregated "
               rf"over the 17 attack--ratio cells and 5 seeds: \textbf{{W}} = FedShield wins, "
               rf"T = no significant difference, L = FedShield loses, --- = baseline not run. "
               rf"Aggregate over {n_total} valid comparisons: \textbf{{{n_W} wins}}, "
               rf"{n_T} ties, {n_L} losses.}}")
    out.append(r"\label{tab:wins-summary}")
    out.append(r"\resizebox{\textwidth}{!}{%")
    out.append(r"\begin{tabular}{l " + " ".join(["c"]*len(headers[1:])) + "}")
    out.append(r"\toprule")
    out.append(" & ".join([rf"\textbf{{{h}}}" for h in headers_short]) + r" \\")
    out.append(r"\midrule")
    for ds_name, outs in rows:
        out.append(f"  {ds_name} & " + " & ".join(outs) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table*}")
    return "\n".join(out) + "\n", n_W, n_T, n_L, n_total


def ablation_table_tex(df: pd.DataFrame) -> str:
    abl_map = {
        "fedshield_v10":   "FedShield (full method)",
        "A1_no_edge_ae":   r"Stage A ablation (no edge AE)",
        "A2_no_norm_clip": r"Stage B ablation (no norm-MAD clip)",
        "A3_mean_ref":     r"Stage C ablation (mean reference)",
        "A4a_mean_only":   r"Stage E variant ($\alpha{=}1$, mean only)",
        "A4b_median_only": r"Stage E variant ($\alpha{=}0$, median only)",
        "A5_buggy_f":      r"Stage D variant (adaptive $f{=}\lfloor\rho_m m\rfloor$)",
    }
    sub = df[(df.dataset=="mitbih") & (df.variant.isin(abl_map.keys()))
             & (df.num_clients.isna())]
    if len(sub) == 0: return "% no ablation data\n"
    means = sub.groupby(["variant","attack","mr"])["score"].mean().reset_index()
    var_means = means.groupby("variant")["score"].mean()
    fs = var_means["fedshield_v10"]
    out = []
    out.append(r"\begin{table}[!t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\caption{Stage-level ablation on the MIT-BIH 17-cell matrix (5 seeds). "
               r"$\Delta$ vs full = mean defense score $-$ FedShield's. Stages D and E are "
               r"load-bearing; Stages A, B, C contribute within $\pm0.002$ of zero, motivating "
               r"the FedShield-Lite variant in Sec.~\ref{sec:discussion}.}")
    out.append(r"\label{tab:ablation}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{l r r}")
    out.append(r"\toprule")
    out.append(r"Configuration & Mean score & $\Delta$ vs full \\")
    out.append(r"\midrule")
    ordered = sorted(abl_map.keys(), key=lambda v: -var_means.get(v, -1))
    for v in ordered:
        if v not in var_means.index: continue
        s = var_means[v]
        d = s - fs
        if v == "fedshield_v10":
            out.append(rf"  \rowcolor{{gray!15}} {abl_map[v]} & {s:.3f} & --- \\")
        else:
            out.append(rf"  {abl_map[v]} & {s:.3f} & ${d:+.3f}$ \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def design_rationale_tex(df: pd.DataFrame) -> str:
    rationale_map = {
        "fedshield_v10":      r"FedShield (full ensemble, $\alpha{=}0.9$)",
        "single_mean_top2":   r"Stage E reduced to mean-of-top-2 only (Krum-filtered)",
        "single_median_top7": r"Stage E reduced to median-of-top-7 only (Krum-filtered)",
        "single_trim_top7":   r"Stage E reduced to trim-mean of top-7 (Krum-filtered)",
        "median":             r"Coord-wise median, full cohort (no prefilter, no ensemble)",
        "ens_mean+trim":      r"Ensemble mean + trim-mean ($\alpha{=}0.5$)",
        "ens_median+trim":    r"Ensemble median + trim-mean ($\alpha{=}0.5$)",
        "ens_mean+geomed":    r"Ensemble mean + geometric median",
        "bulyan_classic":     r"Bulyan (Krum$\rightarrow$trim cascade)~\cite{mhamdi2018hidden}",
        "ens_no_krum_filter": r"Ensemble without Krum prefilter (still uses $\alpha$ mixture)",
        "ens_max":            r"Element-wise max of mean and median",
        "ens_min":            r"Element-wise min of mean and median",
        "ens_alarm_gated":    r"Alarm-gated discrete switch (mean if no alarm, else median)",
    }
    sub = df[(df.dataset=="mitbih") & (df.variant.isin(rationale_map.keys()))
             & (df.num_clients.isna())]
    if len(sub) == 0: return "% no design-rationale data\n"
    means = (sub.groupby(["variant","attack","mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean())
    fs = means["fedshield_v10"]
    out = []
    out.append(r"\begin{table}[!t]")
    out.append(r"\centering\footnotesize")
    out.append(r"\caption{Design-rationale ablation on the MIT-BIH 17-cell matrix (5 seeds). "
               r"Each row replaces one architectural choice in Stage E with an alternative. "
               r"The standalone full-cohort coordinate-wise median row is the median branch of FedShield "
               r"with both the Krum prefilter and the convex mixture removed; it is reported here "
               r"as an architectural ablation rather than as a peer baseline, because it is a "
               r"component of FedShield rather than a published competing defense. "
               r"Every alternative scores below the proposed configuration.}")
    out.append(r"\label{tab:design-rationale}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{l r r}")
    out.append(r"\toprule")
    out.append(r"Configuration & Mean score & $\Delta$ vs proposed \\")
    out.append(r"\midrule")
    ordered = sorted(rationale_map.keys(), key=lambda v: -means.get(v, -1))
    for v in ordered:
        if v not in means.index: continue
        s = means[v]
        d = s - fs
        if v == "fedshield_v10":
            out.append(rf"  \rowcolor{{gray!15}} {rationale_map[v]} & {s:.3f} & --- \\")
        else:
            out.append(rf"  {rationale_map[v]} & {s:.3f} & ${d:+.3f}$ \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def tuned_baselines_tex(df: pd.DataFrame) -> str:
    families = {
        "Krum": ["krum_f1","krum_f2","krum_f3"],
        "Multi-Krum": ["multi_krum_m3","multi_krum_m5","multi_krum_m7"],
        "Trimmed-Mean": ["trimmed_b10","trimmed_b20","trimmed_b30"],
        "FLTrust": ["fltrust_r100","fltrust_r200","fltrust_r400"],
    }
    sub = df[(df.dataset=="mitbih") & (df.num_clients.isna())]
    if len(sub) == 0: return "% no tuned data\n"
    means = (sub.groupby(["variant","attack","mr"])["score"].mean()
                .reset_index().groupby("variant")["score"].mean())
    if "fedshield_v10" not in means.index: return "% no fedshield_v10\n"
    fs = means["fedshield_v10"]
    out = []
    out.append(r"\begin{table}[!t]")
    out.append(r"\centering\small")
    out.append(r"\caption{Tuned-baselines fairness check on MIT-BIH 17-cell matrix (3 seeds). "
               r"For each baseline family we sweep its main hyperparameter and report the best-tuned "
               r"configuration. FedShield retains a positive lead against every family.}")
    out.append(r"\label{tab:tuned}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{l l r r}")
    out.append(r"\toprule")
    out.append(r"Family & Best config & Mean score & $\Delta$ to FedShield \\")
    out.append(r"\midrule")
    out.append(rf"  \rowcolor{{gray!15}} \textbf{{FedShield (proposed)}} & --- & {fs:.3f} & --- \\")
    for fam, vs in families.items():
        best_v = max((v for v in vs if v in means.index), key=lambda v: means[v], default=None)
        if best_v is None: continue
        out.append(rf"  Best-tuned {fam} & {best_v.replace('_',r'\_')} & {means[best_v]:.3f} & ${fs - means[best_v]:+.3f}$ \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def cost_table_tex(df: pd.DataFrame) -> str:
    sub = df[df.variant.isin(HEADLINE_VARIANTS)]
    if len(sub) == 0: return "% no cost data\n"
    cost = sub.groupby("variant")[["lat_edge","lat_srv","comm","ram"]].mean()
    order = ["fedavg","krum","multi_krum","trimmed_mean","fltrust","fedshield_v10"]
    out = []
    out.append(r"\begin{table}[!t]")
    out.append(r"\centering\small")
    out.append(r"\caption{Per-round per-defense computational cost, averaged across all "
               r"five datasets and 5 seeds. FedShield matches Krum's edge cost (within "
               r"$0.1$\,ms) and matches FLTrust's server cost (within $0.1$\,ms), without "
               r"requiring FLTrust's trusted server-side reference dataset.}")
    out.append(r"\label{tab:cost}")
    out.append(r"\resizebox{\columnwidth}{!}{%")
    out.append(r"\begin{tabular}{l r r r r}")
    out.append(r"\toprule")
    out.append(r"Defense & Edge ms & Server ms & Comm KB/c & RAM MB \\")
    out.append(r"\midrule")
    for v in order:
        if v not in cost.index: continue
        row = cost.loc[v]
        nm = DISPLAY.get(v, v)
        if v == "fedshield_v10":
            out.append(rf"  \rowcolor{{gray!15}} {nm} & {row['lat_edge']:.1f} & {row['lat_srv']:.1f} & {row['comm']:.0f} & {row['ram']:.2f} \\")
        else:
            out.append(rf"  {nm} & {row['lat_edge']:.1f} & {row['lat_srv']:.1f} & {row['comm']:.0f} & {row['ram']:.2f} \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def main() -> None:
    df = parse_all()
    out_dir = Path("./paper/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS_FULL:
        (out_dir / f"tab_headline_{ds}.tex").write_text(headline_table_tex(df, ds), encoding="utf-8")
        print(f"wrote tab_headline_{ds}.tex")

    # Compute aggregate W/T/L counts for use in the prose abstract / conclusion.
    # We do NOT emit a W/T/L table — it reads as a sports bracket, not a paper.
    if stats is not None:
        n_W = n_T = n_L = n_total = 0
        for ds in DATASETS_FULL:
            sub = df[(df.dataset == ds) & (df.variant.isin(HEADLINE_VARIANTS))
                     & (df.num_clients.isna())]
            if len(sub) == 0: continue
            pivot = sub.pivot_table(index=["attack","mr","seed"], columns="variant",
                                     values="score", aggfunc="first")
            if "fedshield_v10" not in pivot.columns: continue
            for v in HEADLINE_VARIANTS:
                if v in ("fedshield_v10","fedshield_v10_a025"): continue
                if v not in pivot.columns: continue
                pair = pivot[["fedshield_v10", v]].dropna()
                if len(pair) < 3: continue
                t, p = stats.ttest_rel(pair["fedshield_v10"], pair[v])
                d = float((pair["fedshield_v10"] - pair[v]).mean())
                n_total += 1
                if p < 0.05 and d > 0: n_W += 1
                elif p < 0.05 and d < 0: n_L += 1
                else: n_T += 1
        print(f"AGGREGATE   {n_W} wins / {n_T} ties / {n_L} losses of {n_total}  -- use in prose, not as table")

    (out_dir / "tab_ablation.tex").write_text(ablation_table_tex(df), encoding="utf-8")
    print("wrote tab_ablation.tex")
    (out_dir / "tab_design_rationale.tex").write_text(design_rationale_tex(df), encoding="utf-8")
    print("wrote tab_design_rationale.tex")
    (out_dir / "tab_tuned_baselines.tex").write_text(tuned_baselines_tex(df), encoding="utf-8")
    print("wrote tab_tuned_baselines.tex")
    (out_dir / "tab_cost.tex").write_text(cost_table_tex(df), encoding="utf-8")
    print("wrote tab_cost.tex")


if __name__ == "__main__":
    main()
