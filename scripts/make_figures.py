"""Single source of truth for all paper figures and tables.

Produces five figures and three tables, all from the same parsed dataframe,
with consistent styling. Replaces the older make_paper_figures.py,
make_asr_breakdown.py, and make_per_attack_breakdown.py.

Main paper figures (all referenced from results_section.tex):
  fig_overview.pdf            pipeline diagram (kept as-is, regenerated here)
  fig_headline.pdf            2x4 grid: ASR (top) and defense score (bottom)
                              per method per dataset, winner ringed
  fig_consistency.pdf         top-3 placement count per method across all
                              (dataset, metric) cells
  fig_attack_interaction.pdf  rho-sweep on scaling vs sign_flip across
                              all four datasets
Appendix figure:
  fig_per_attack_supplement.pdf
                              4x8 heatmap of FedShield rank per
                              (dataset, attack), single figure replacing
                              the previous eight per-attack figures

Main paper tables:
  tab_headline.tex            single consolidated table: rows = methods,
                              cols = (Acc / ASR / Score) x 4 datasets,
                              with Wilcoxon significance against
                              FedShield-Mean
  tab_consistency.tex         top-1 / top-3 placement counts per method
Appendix table:
  tab_per_attack_supplement.tex
                              compact 64-row table: (dataset, attack, rho_m)
                              vs methods, defense_score with winner bolded

Run:
  python -m scripts.make_figures
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


KNOWN_ATTACKS = [
    "label_flip+backdoor+sign_flip+scaling",
    "sign_flip+scaling", "backdoor+scaling",
    "sign_flip", "label_flip", "backdoor", "scaling", "noise_update", "clean",
]
_ATTACK_ALT = "|".join(re.escape(a) for a in KNOWN_ATTACKS)
RE = re.compile(
    rf"^proto_(?P<dataset>[^_]+)_(?P<variant>.+?)_"
    rf"(?P<attack>{_ATTACK_ALT})_r(?P<mr>\d{{2}})_s(?P<seed>\d{{2}})"
    rf"(?:_n(?P<n>\d{{2}}))?_metrics\.csv$"
)
VARIANTS = {
    "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum",
    "trimmed_mean": "Trim-Mean",
    "rfa": "RFA",
    "dnc": "DnC",
    "fltrust": "FLTrust",
    "fedshield_v10_a025": "FedShield",
    "fedshield_v10": "FedShield-Mean",
}
VARIANTS_SHORT = {
    "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "M-Krum",
    "trimmed_mean": "Trim",
    "rfa": "RFA",
    "dnc": "DnC",
    "fltrust": "FLTr",
    "fedshield_v10_a025": "FS",
    "fedshield_v10": "FS-M",
}
DATASETS = {
    "mitbih": "MIT-BIH", "ciciomt": "CIC-IoMT-2024",
    "ptbxl": "PTB-XL", "physionet2017": "PhysioNet/CinC 2017",
}
ATTACK_DISPLAY = {
    "scaling":                                "scaling",
    "backdoor+scaling":                       "backdoor+scale",
    "sign_flip+scaling":                      "sign+scale",
    "label_flip+backdoor+sign_flip+scaling":  "full composite",
    "noise_update":                           "Gaussian noise",
    "sign_flip":                              "sign flip",
    "backdoor":                               "backdoor",
    "label_flip":                             "label flip",
}
ATTACK_ORDER = list(ATTACK_DISPLAY.keys())
METHOD_ORDER = list(VARIANTS.keys())

OUR = "fedshield_v10_a025"      # FedShield (main, alpha=0.25)
OUR_VARIANT = "fedshield_v10"   # FedShield-Mean (alpha=0.90)
COLOR_FS = "#c0392b"            # FedShield deep red (main method)
COLOR_VAR = "#e67e22"           # FedShield-Mean orange (variant)
COLOR_BL = "#34495e"            # baselines slate
COLOR_RING = "#f1c40f"          # gold winner ring


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    rows = []
    for csv in Path("results/proto").rglob("proto_*_metrics.csv"):
        m = RE.match(csv.name)
        if not m or m.group("n"):
            continue
        if m.group("variant") not in VARIANTS or m.group("dataset") not in DATASETS:
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if len(df) == 0:
            continue
        last = df.iloc[-1]
        acc = float(last.get("acc", float("nan")))
        f1  = float(last.get("f1",  float("nan")))
        asr = float(last.get("asr", float("nan")))
        rows.append({
            "dataset": m.group("dataset"),
            "variant": m.group("variant"),
            "attack":  m.group("attack"),
            "mr":      round(float(m.group("mr")) / 100, 2),
            "seed":    int(m.group("seed")),
            "acc":     acc,
            "f1":      f1,
            "asr":     asr,
            # Score = F1 * (1 - ASR) -- F1-based to handle majority-class
            # collapse on imbalanced healthcare-IoT data; acc-based score
            # is retained as 'score_acc' for compatibility with older
            # tables. The headline score is f1-based.
            "score":     f1 * (1.0 - asr),
            "score_acc": acc * (1.0 - asr),
        })
    df = pd.DataFrame(rows)
    return df


def style_color(v: str) -> str:
    if v == OUR:
        return COLOR_FS
    if v == OUR_VARIANT:
        return COLOR_VAR
    return COLOR_BL


# ---------------------------------------------------------------------------
# Figure 1 (pipeline overview) — text-only diagram, kept simple
# ---------------------------------------------------------------------------

def fig_overview(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    ax.axis("off")
    stages = [
        ("Stage A", "Anomaly\ndetection",  "client", "#e9eef7"),
        ("Stage B", "Magnitude\nfilter",   "server", "#e9eef7"),
        ("Stage C", "Direction\nreference", "server", "#e9eef7"),
        ("Stage D", "Robust\nselection",   "server", "#e9eef7"),
        ("Stage E", "Ensemble\naggregator", "server", "#f5b7b1"),
    ]
    n = len(stages); box_w = 0.16; gap = 0.025
    total_w = n * box_w + (n - 1) * gap
    x0 = (1.0 - total_w) / 2.0
    for i, (label, role, side, color) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        ax.add_patch(plt.Rectangle((x, 0.30), box_w, 0.50,
                                    facecolor=color, edgecolor="black", linewidth=1.0))
        ax.text(x + box_w/2, 0.74, label, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + box_w/2, 0.55, role, ha="center", va="center", fontsize=9)
        ax.text(x + box_w/2, 0.20, side, ha="center", va="center", fontsize=8, style="italic", color="#555")
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap - 0.003, 0.55),
                        xytext=(x + box_w + 0.003, 0.55),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))
    x_E = x0 + (n - 1) * (box_w + gap)
    ax.text(x_E + box_w/2, 0.92, "proposed", ha="center", va="center",
            fontsize=8, style="italic", color="#a93226")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2 (headline) — combined ASR + score, 2 rows x 4 cols
# ---------------------------------------------------------------------------

def fig_headline(df: pd.DataFrame, out: Path) -> None:
    rows_meta = [
        ("asr",   "Attack-effect term $s$ (lower better)", True),
        ("score", "Defense score (higher better)",     False),
    ]
    fig, axes = plt.subplots(2, len(DATASETS),
                              figsize=(3.0 * len(DATASETS), 6.5),
                              sharey="row")
    for r_idx, (metric, ylabel, lower) in enumerate(rows_meta):
        for c_idx, ds in enumerate(DATASETS):
            ax = axes[r_idx, c_idx]
            sub = df[df.dataset == ds]
            # Sort by value: winner at top, worst at bottom. barh draws the
            # first label at y=0 (bottom), so put the worst entry first.
            means = sub.groupby("variant")[metric].mean().sort_values(
                ascending=not lower)
            order = means.index.tolist()
            labels = [VARIANTS_SHORT[v] for v in order]
            vals = [means[v] for v in order]
            colors = [style_color(v) for v in order]
            bars = ax.barh(labels, vals, color=colors,
                           edgecolor="black", linewidth=0.3)
            if r_idx == 0:
                ax.set_title(DATASETS[ds], fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            # Annotate every bar with its value so the value order is
            # visible at a glance.
            xmax = max(vals) if vals else 1.0
            for bar, v, val in zip(bars, order, vals):
                weight = "bold" if v in (OUR, OUR_VARIANT) else "normal"
                tcolor = bar.get_facecolor() if v in (OUR, OUR_VARIANT) else "#222"
                ax.text(val + xmax * 0.015,
                        bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=7,
                        color=tcolor, fontweight=weight)
            # Extend x-axis so labels fit
            ax.set_xlim(0, xmax * 1.2)
    fig.suptitle("Headline performance: per-method attack-effect term $s$ (top) and "
                 "defense score (bottom) per dataset (5 seeds, 32 attacked cells)",
                 fontsize=11, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3 (consistency) — top-3 placement count per method
# ---------------------------------------------------------------------------

def fig_consistency(df: pd.DataFrame, out: Path) -> None:
    """For each method, count how often it ranks in the top-3 across the
    12 (dataset, metric) cells, where metric in {f1, asr, score}."""
    counts = {}
    for v in METHOD_ORDER:
        top1 = top2 = top3 = 0
        worst = 0
        for ds in DATASETS:
            for metric, lower in [("f1", False), ("asr", True), ("score", False)]:
                ag = (df.groupby(["dataset", "variant"])[metric].mean()
                        .unstack("variant"))
                s = ag.loc[ds].sort_values(ascending=lower)
                rank = list(s.index).index(v) + 1
                if rank == 1: top1 += 1
                if rank <= 2: top2 += 1
                if rank <= 3: top3 += 1
                if rank > worst: worst = rank
        counts[v] = (top1, top2, top3, worst)
    # Sort by top3 desc, then top1 desc
    ordered = sorted(METHOD_ORDER,
                     key=lambda v: (-counts[v][2], -counts[v][0], counts[v][3]))
    labels = [VARIANTS[v] for v in ordered]
    top1_vals = [counts[v][0] for v in ordered]
    top3_vals = [counts[v][2] for v in ordered]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y_pos = np.arange(len(ordered))
    # Background bar = top-3 count
    ax.barh(y_pos, top3_vals, color="#bdc3c7",
            edgecolor="black", linewidth=0.3, label="Top-3 placements")
    # Foreground = top-1 count
    ax.barh(y_pos, top1_vals,
            color=[style_color(v) for v in ordered],
            edgecolor="black", linewidth=0.3, label="Rank-1 placements")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Placements out of 12 (dataset x metric) cells")
    ax.set_xlim(0, 12)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)
    # Annotate worst rank to the right of each bar
    for i, v in enumerate(ordered):
        t1, _, t3, worst = counts[v]
        ax.text(t3 + 0.15, i,
                f"top-3: {t3}/12   worst rank: {worst}",
                va="center", fontsize=8, color="#444")
    ax.set_title("Cross-(dataset x metric) consistency: how often each\n"
                 "method places in the top 3 (lower is worse). Methods ordered\n"
                 "by top-3 count.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 4 (attack interaction) — scaling vs sign_flip rho-sweep
# ---------------------------------------------------------------------------

def fig_attack_interaction(df: pd.DataFrame, out: Path) -> None:
    attacks = [("scaling",   "scaling (delta -> 10*delta, magnitude)"),
               ("sign_flip", "sign flip (delta -> -5*delta, direction)")]
    fig, axes = plt.subplots(2, len(DATASETS),
                              figsize=(3.0 * len(DATASETS), 6.0),
                              sharey="row")
    cmap = plt.cm.tab10
    handles = {}
    for r_idx, (atk, atk_label) in enumerate(attacks):
        for c_idx, ds in enumerate(DATASETS):
            ax = axes[r_idx, c_idx]
            sub = df[(df.dataset == ds) & (df.attack == atk)]
            if len(sub) == 0:
                ax.set_visible(False); continue
            means = (sub.groupby(["mr", "variant"])["score"].mean()
                        .unstack("variant").sort_index())
            stds = (sub.groupby(["mr", "variant"])["score"].std()
                       .unstack("variant").sort_index().reindex(means.index).fillna(0.0))
            order = [v for v in METHOD_ORDER if v in means.columns
                     and v not in (OUR, OUR_VARIANT)]
            if OUR_VARIANT in means.columns: order.append(OUR_VARIANT)
            if OUR in means.columns: order.append(OUR)
            for j, v in enumerate(order):
                is_ours = v == OUR
                is_pd = v == OUR_VARIANT
                color = (COLOR_FS if is_ours
                         else COLOR_VAR if is_pd else cmap(j % 10))
                lw = 2.4 if is_ours else (2.0 if is_pd else 1.0)
                ms = 7 if is_ours else (6 if is_pd else 4)
                marker = "*" if is_ours else ("D" if is_pd else "o")
                alpha = 1.0 if (is_ours or is_pd) else 0.6
                z = 10 if is_ours else (8 if is_pd else 5)
                band_alpha = 0.22 if (is_ours or is_pd) else 0.07
                ax.fill_between(means.index,
                                means[v] - stds[v], means[v] + stds[v],
                                color=color, alpha=band_alpha, linewidth=0,
                                zorder=z - 1)
                line, = ax.plot(means.index, means[v], marker=marker, color=color,
                                linewidth=lw, markersize=ms, alpha=alpha, zorder=z)
                handles.setdefault(v, line)
            if r_idx == 0:
                ax.set_title(DATASETS[ds], fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(f"Defense score\n({atk_label})", fontsize=9)
            if r_idx == len(attacks) - 1:
                ax.set_xlabel("rho_m", fontsize=9)
            ax.set_xticks([0.1, 0.2, 0.3, 0.4])
            ax.tick_params(labelsize=7)
            ax.grid(linestyle=":", alpha=0.4)
    handles_list = [handles[v] for v in METHOD_ORDER if v in handles]
    labels_list = [VARIANTS[v] for v in METHOD_ORDER if v in handles]
    fig.legend(handles_list, labels_list, loc="lower center", ncol=8,
               fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Attack-type x partition interaction: defense score "
                 "vs rho_m for a magnitude attack and a direction attack "
                 "(5 seeds)",
                 fontsize=11, y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Appendix figure (per-attack supplement) — rank heatmap
# ---------------------------------------------------------------------------

def fig_per_attack_supplement(df: pd.DataFrame, out: Path) -> None:
    """4 datasets x 8 attacks heatmap of FedShield's defense_score rank
    (1 = best). Compact replacement for the previous 8 per-attack figures."""
    M = np.full((len(ATTACK_ORDER), len(DATASETS)), np.nan)
    M_default = np.full_like(M, np.nan)
    for j, ds in enumerate(DATASETS):
        for i, atk in enumerate(ATTACK_ORDER):
            sub = df[(df.dataset == ds) & (df.attack == atk)]
            if len(sub) == 0: continue
            means = sub.groupby("variant")["score"].mean().sort_values(ascending=False)
            order = means.index.tolist()
            try:
                M[i, j] = order.index(OUR_VARIANT) + 1
                M_default[i, j] = order.index(OUR) + 1
            except ValueError:
                pass
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    fig.subplots_adjust(wspace=0.15)
    titles = [("FedShield (alpha=0.25) rank", M),
              ("FedShield-Mean (alpha=0.90) rank", M_default)]
    for idx, (ax, (title, mat)) in enumerate(zip(axes, titles)):
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=8)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DATASETS[d] for d in DATASETS],
                            rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(ATTACK_ORDER)))
        if idx == 0:
            ax.set_yticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], fontsize=9)
        for i in range(len(ATTACK_ORDER)):
            for j in range(len(DATASETS)):
                v = mat[i, j]
                if np.isnan(v): continue
                color = "white" if v >= 6 or v <= 2 else "black"
                weight = "bold" if v <= 2 else "normal"
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        fontsize=11, color=color, fontweight=weight)
        ax.set_title(title, fontsize=10, pad=8)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03, shrink=0.85)
    cbar.set_label("Rank (1 = best, 8 = worst)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def tab_headline(df: pd.DataFrame, out: Path) -> None:
    """Single consolidated headline table.

    Layout: rows = methods, columns = (Acc, ASR, Score) x 4 datasets = 12.
    Per-dataset winner per metric is in bold. Wilcoxon significance against
    FedShield (alpha=0.25, the default configuration, OUR) is reported on
    the score column only.
    """
    from scipy.stats import wilcoxon
    cells = {}
    for v in METHOD_ORDER:
        sub = df[df.variant == v]
        for ds in DATASETS:
            sub_ds = sub[sub.dataset == ds]
            for metric in ("f1", "asr", "score"):
                cells[(v, ds, metric)] = sub_ds[metric].mean()

    # Significance for score vs FedShield-Mean per dataset
    sig = {}
    for ds in DATASETS:
        fs_sub = (df[(df.variant == OUR) & (df.dataset == ds)]
                    .sort_values(["attack", "mr", "seed"]).reset_index(drop=True))
        for v in METHOD_ORDER:
            if v == OUR:
                sig[(v, ds)] = None
                continue
            v_sub = (df[(df.variant == v) & (df.dataset == ds)]
                       .sort_values(["attack", "mr", "seed"]).reset_index(drop=True))
            n = min(len(fs_sub), len(v_sub))
            try:
                _, p = wilcoxon(v_sub["score"].values[:n], fs_sub["score"].values[:n])
            except Exception:
                p = float("nan")
            sig[(v, ds)] = p

    short = {
        "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum",
        "trimmed_mean": "Trim-Mean", "fltrust": "FLTrust",
        "fedshield_v10": "\\textbf{FedShield-Mean}",
        "fedshield_v10_a025": "\\textbf{FedShield}",
    }

    # Determine winner per (dataset, metric)
    winners = {}
    for ds in DATASETS:
        for metric, lower in [("f1", False), ("asr", True), ("score", False)]:
            vals = {v: cells[(v, ds, metric)] for v in METHOD_ORDER}
            best_v = min(vals, key=vals.get) if lower else max(vals, key=vals.get)
            winners[(ds, metric)] = best_v

    def fmt_score(v, ds, val):
        s = f"{val:.3f}"
        if winners[(ds, "score")] == v:
            # Highlight via cell color and bold for double-emphasis
            return "\\cellcolor{gray!20}\\textbf{" + s + "}"
        return s

    def fmt_plain(v, ds, metric, val):
        s = f"{val:.3f}"
        if winners[(ds, metric)] == v:
            return "\\cellcolor{gray!20}\\textbf{" + s + "}"
        return s

    # Build header (multi-row for dataset + metric)
    ds_header_cells = []
    for ds in DATASETS:
        ds_header_cells.append("\\multicolumn{3}{c}{" + DATASETS[ds] + "}")
    ds_header = "Method & " + " & ".join(ds_header_cells) + " \\\\"
    cmid_parts = []
    col = 2
    for _ in DATASETS:
        cmid_parts.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    cmid_line = " ".join(cmid_parts)
    metric_header = " & ".join(["", *(["F1", "ASR", "Score"] * len(DATASETS))]) + " \\\\"

    body_lines = []
    for v in METHOD_ORDER:
        cells_str = []
        for ds in DATASETS:
            cells_str.append(fmt_plain(v, ds, "f1", cells[(v, ds, "f1")]))
            cells_str.append(fmt_plain(v, ds, "asr", cells[(v, ds, "asr")]))
            cells_str.append(fmt_score(v, ds, cells[(v, ds, "score")]))
        body_lines.append(short[v] + " & " + " & ".join(cells_str) + " \\\\")

    col_spec = "l" + "rrr" * len(DATASETS)
    tex = (
        "\\begin{table*}[!t]\n\\centering\\footnotesize\n"
        "\\caption{Headline performance per method per dataset, averaged "
        "over 32 attacked cells (8 attacks $\\times$ 4 ratios) and 5 seeds. "
        "F1: macro-averaged F1 score, used in place of clean accuracy to "
        "penalize majority-class collapse on the imbalanced healthcare-IoT "
        "datasets (MIT-BIH is $\\sim$83\\% Normal beats; CIC-IoMT-2024 has "
        "skewed protocol-class distribution). ASR: attack success rate. "
        "Score $= \\mathrm{F1}\\cdot(1-\\mathrm{ASR})$ is computed for "
        "each matched attack-ratio-seed cell and then averaged; the "
        "reported mean Score is therefore not necessarily equal to the "
        "product of the reported mean F1 and $1-$ the reported mean "
        "ASR. Best result per column is highlighted.}\n"
        "\\label{tab:headline}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
        f"{ds_header}\n{cmid_line}\n{metric_header}\n\\midrule\n"
        + "\n".join(body_lines)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
    )
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")


def tab_consistency(df: pd.DataFrame, out: Path) -> None:
    """Top-1 / top-3 placement counts per method across (dataset, metric) cells."""
    counts = {}
    for v in METHOD_ORDER:
        top1 = top3 = 0
        worst = 0
        for ds in DATASETS:
            for metric, lower in [("f1", False), ("asr", True), ("score", False)]:
                ag = (df.groupby(["dataset", "variant"])[metric].mean()
                        .unstack("variant"))
                s = ag.loc[ds].sort_values(ascending=lower)
                rank = list(s.index).index(v) + 1
                if rank == 1: top1 += 1
                if rank <= 3: top3 += 1
                if rank > worst: worst = rank
        counts[v] = (top1, top3, worst)
    ordered = sorted(METHOD_ORDER,
                     key=lambda v: (-counts[v][1], -counts[v][0], counts[v][2]))

    short = {
        "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum",
        "trimmed_mean": "Trim-Mean", "fltrust": "FLTrust",
        "fedshield_v10": "\\textbf{FedShield-Mean}",
        "fedshield_v10_a025": "\\textbf{FedShield}",
    }
    body = []
    for v in ordered:
        t1, t3, w = counts[v]
        body.append(f"{short[v]} & {t1} & {t3} & {w} \\\\")
    tex = (
        "\\begin{table}[!t]\n\\centering\\footnotesize\n"
        "\\caption{Cross-(dataset $\\times$ metric) consistency. Each method "
        "is ranked on 12 cells: 4 datasets $\\times$ \\{F1, ASR, Score\\}. "
        "The table reports how often each method places at rank 1 or in the "
        "top 3, and the worst rank it ever achieves across all 12 cells. "
        "FedShield is the only method that places in the top 3 on at "
        "least 80\\% of cells with a worst-case rank below 8.}\n"
        "\\label{tab:consistency}\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Method & Top-1 / 12 & Top-3 / 12 & Worst rank \\\\\n\\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")


def fig_decision_guide(out: Path) -> None:
    """Deployment configuration-selection flowchart.

    Academic top-tier style: minimal palette, proper decision diamonds,
    one accent color (recommended-default path), clean Manhattan routing.
    """
    from matplotlib.patches import FancyBboxPatch, Polygon
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 12.5)

    # Palette: near-black borders, light neutral fills, one accent for the
    # recommended-default terminal. Avoids the rainbow-flowchart look.
    INK         = "#1f2937"   # near-black borders and arrows
    FILL_STEP   = "#f3f4f6"   # very light grey for process steps
    FILL_DECIDE = "#ffffff"   # white for decision diamonds
    FILL_TERM   = "#f9fafb"   # off-white for terminals
    FILL_REC    = "#fde7e7"   # subtle warm tint for the recommended default
    EC_REC      = "#9c2b2b"   # accent border for the recommended default
    FILL_WARN   = "#f5f5f5"   # neutral grey for the warn terminal
    GREY_DASH   = "#6b7280"   # dashed-border colour for warn

    LW_BOX, LW_DECIDE, LW_REC = 1.0, 1.1, 1.6
    FS_STEP, FS_DECIDE, FS_TERM, FS_LABEL = 10, 10, 10, 9

    def rect(x, y, w, h, txt, *, fill=FILL_STEP, edge=INK, lw=LW_BOX,
             bold=False, fs=FS_STEP):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h, boxstyle="round,pad=0.04",
            facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=fs, fontweight=("bold" if bold else "normal"),
                color=INK, zorder=3)

    def diamond(x, y, w, h, txt, *, fs=FS_DECIDE):
        pts = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=FILL_DECIDE,
                             edgecolor=INK, linewidth=LW_DECIDE, zorder=2))
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)

    def arrow(x1, y1, x2, y2, *, label=None, label_pos="right", color=INK,
              lw=1.1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    mutation_scale=12, shrinkA=0, shrinkB=0),
                    zorder=1)
        if label is not None:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.22
            if label_pos == "right":
                ax.text(mx + offset, my, label, ha="left", va="center",
                        fontsize=FS_LABEL, color=INK)
            elif label_pos == "left":
                ax.text(mx - offset, my, label, ha="right", va="center",
                        fontsize=FS_LABEL, color=INK)
            elif label_pos == "above":
                ax.text(mx, my + offset, label, ha="center", va="bottom",
                        fontsize=FS_LABEL, color=INK)

    def L_arrow(x1, y1, x_turn, y2, x2, *, label=None, color=INK, lw=1.1):
        ax.plot([x1, x_turn], [y1, y1], color=color, lw=lw, zorder=1,
                solid_capstyle="round")
        ax.annotate("", xy=(x2, y2), xytext=(x_turn, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                    mutation_scale=12, shrinkA=0, shrinkB=0),
                    zorder=1)
        if label is not None:
            ax.text((x1 + x_turn) / 2, y1 + 0.25, label, ha="center",
                    va="bottom", fontsize=FS_LABEL, color=INK)

    # --- Row 1: calibration step ----------------------------------------
    rect(7, 11.5, 8.6, 1.0,
         "Calibration round (no attack) and stress-test subset:\n"
         "characterize honest-update geometry",
         fs=FS_STEP)
    arrow(7, 11.0, 7, 10.3)

    # --- Row 2: decision 1 (cohort size) --------------------------------
    diamond(7, 9.3, 4.6, 1.8, "Cohort size m >= 7 ?")

    # YES (down to decision 2) and NO (left then down to warn)
    arrow(7, 8.4, 7, 7.7, label="yes", label_pos="right")
    L_arrow(4.7, 9.3, 2.0, 6.8, 2.0, label="no")

    # --- Warn terminal (small cohort) -----------------------------------
    ax.add_patch(FancyBboxPatch(
        (2.0 - 1.8, 6.2 - 0.85), 3.6, 1.7, boxstyle="round,pad=0.04",
        facecolor=FILL_WARN, edgecolor=GREY_DASH, linewidth=1.0,
        linestyle="--", zorder=2))
    ax.text(2.0, 6.2,
            "Use adaptive\nsmall-cohort variant\nor baseline median",
            ha="center", va="center", fontsize=FS_TERM, color=INK,
            fontstyle="italic", zorder=3)

    # --- Row 3: decision 2 (dispersion) ---------------------------------
    diamond(7, 6.8, 6.8, 1.8,
            "Calibration or stress-test indicates\ncentroid-near (mimicry-style) geometry ?",
            fs=FS_DECIDE - 1)

    # Branches down to result terminals
    arrow(5.0, 5.9, 5.0, 4.0, label="yes", label_pos="left")
    arrow(9.0, 5.9, 9.0, 4.0, label="no /\nunknown", label_pos="right")

    # --- Row 4: terminal result boxes -----------------------------------
    rect(5.0, 3.2, 3.6, 1.5,
         "FedShield-Mean\n(alpha = 0.90)\n"
         "centroid-near /\nmimicry-style regime",
         fill=FILL_TERM, lw=1.1, bold=False, fs=FS_TERM)
    rect(9.0, 3.2, 3.6, 1.5,
         "FedShield\n(alpha = 0.25)\n"
         "recommended default",
         fill=FILL_REC, edge=EC_REC, lw=LW_REC, bold=True, fs=FS_TERM)

    # --- Legend ---------------------------------------------------------
    # Small legend showing what a thick-bordered red box means.
    leg_y = 1.4
    ax.add_patch(FancyBboxPatch((0.6, leg_y - 0.25), 0.5, 0.5,
                                 boxstyle="round,pad=0.04",
                                 facecolor=FILL_REC, edgecolor=EC_REC,
                                 linewidth=LW_REC, zorder=2))
    ax.text(1.35, leg_y, "recommended default configuration",
            ha="left", va="center", fontsize=FS_LABEL, color=INK)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def fig_dispersion(df: pd.DataFrame, df_clean: pd.DataFrame, out: Path) -> None:
    """Operational partition-dispersion proxy vs FedShield advantage.

    D = std of clean-cell defense_score across the five honest baselines
    (FedAvg, Krum, Multi-Krum, Trimmed-Mean, FLTrust), averaged over seeds.
    Higher D means the honest aggregators disagree on the clean data,
    indicating that the client partition produces dispersed honest update
    geometry. We pair D against the FedShield advantage: the difference
    between FedShield's mean defense score and the best baseline's mean
    defense score, on the attacked cells.
    """
    HONEST = ["fedavg", "krum", "multi_krum", "trimmed_mean", "fltrust"]
    pts = []
    for ds in DATASETS:
        clean = df_clean[(df_clean.dataset == ds)
                         & (df_clean.variant.isin(HONEST))]
        method_means = clean.groupby("variant")["score"].mean()
        D_val = float(method_means.std(ddof=0))
        attacked = df[df.dataset == ds]
        by_v = attacked.groupby("variant")["score"].mean()
        fs_val = float(by_v.get(OUR, float("nan")))
        baseline_best = max(by_v[v] for v in HONEST if v in by_v.index)
        adv = fs_val - baseline_best
        pts.append((ds, D_val, adv))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(xs, ys, s=180, c=COLOR_FS, edgecolors="black",
                linewidth=0.8, zorder=5)
    for (ds, x, y) in pts:
        ax.annotate(DATASETS[ds], (x, y),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("Partition dispersion proxy D"
                  " (std of clean-cell defense score across honest baselines)",
                  fontsize=9)
    ax.set_ylabel("FedShield advantage\n(score - best baseline score)",
                  fontsize=9)
    ax.set_xlim(0.02, 0.13)
    ax.grid(linestyle=":", alpha=0.4)
    ax.set_title("Where FedShield helps: operational partition-dispersion view",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def tab_assumptions(out: Path) -> None:
    """Comparison of method assumptions across Byzantine-robust baselines
    and FedShield. Reports: server reference data, cohort-size requirement,
    server-side time complexity, and partition-awareness."""
    rows = [
        ("Krum~\\cite{blanchard2017byzantine}",      "no",  "$m \\geq f+3$",    "$O(m^2 d)$",        "no"),
        ("Multi-Krum~\\cite{blanchard2017byzantine}", "no", "$m \\geq f+3$",    "$O(m^2 d)$",        "no"),
        ("Trimmed-Mean~\\cite{yin2018byzantine}",    "no",  "$m \\geq 2\\beta+1$","$O(m d \\log m)$","no"),
        ("FLTrust~\\cite{cao2021fltrust}",          "\\textbf{yes}", "none",  "$O(m d)$",          "no"),
        ("\\cellcolor{gray!20}\\textbf{FedShield}",  "\\cellcolor{gray!20}no",
         "\\cellcolor{gray!20}$m \\geq 7$",
         "\\cellcolor{gray!20}$O(m^2 d)$",
         "\\cellcolor{gray!20}\\textbf{yes}"),
    ]
    header = (" & ".join(["Method",
                           "Server reference data",
                           "Cohort-size requirement",
                           "Server-side complexity",
                           "Tunable mixing weight"])
              + " \\\\")
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    tex = (
        "\\begin{table}[!t]\n\\centering\\footnotesize\n"
        "\\caption{Comparison of structural assumptions across robust "
        "aggregation methods. The \\emph{Tunable mixing weight} column "
        "indicates whether the method exposes a deployment-time scalar "
        "that can be configured to the observed client-partition "
        "geometry; FedShield's mixing weight $\\alpha$ plays this role.}\n"
        "\\label{tab:assumptions}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        f"{header}\n\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )
    out.write_text(tex, encoding="utf-8")
    print(f"wrote {out}")


def tab_per_attack_supplement(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-dataset granular tables: rows = (attack, rho_m), cols = methods.
    Generates one table per dataset (32 rows each) so that each fits on a
    single page; the previous combined 128-row table overflowed.
    """
    means = (df.groupby(["dataset", "attack", "mr", "variant"])["score"]
                .mean().unstack("variant"))
    asr_means = (df.groupby(["dataset", "attack", "mr", "variant"])["asr"]
                   .mean().unstack("variant"))
    cols = [v for v in METHOD_ORDER if v in means.columns]
    means = means[cols]
    asr_means = asr_means[cols]

    short = {v: VARIANTS_SHORT[v] for v in cols}
    short[OUR] = "\\textbf{FS}"
    short[OUR_VARIANT] = "\\textbf{FS-PD}"
    header = (" & ".join(["Attack", "$\\rho_m$"] + [short[v] for v in cols])
              + " \\\\")
    col_spec = "ll" + "r" * len(cols)

    for ds in DATASETS:
        rows_out = []
        last_atk = None
        keys = sorted(
            [k for k in means.index.tolist() if k[0] == ds],
            key=lambda k: (
                ATTACK_ORDER.index(k[1]) if k[1] in ATTACK_ORDER else 99,
                k[2],
            ),
        )
        for (_ds, atk, mr) in keys:
            row = means.loc[(ds, atk, mr)]
            asr_row = asr_means.loc[(ds, atk, mr)]
            best = row.idxmax()
            cells_str = []
            for v in cols:
                val = row[v]
                asr_val = asr_row[v]
                s = f"{val:.3f}"
                if val < 0.001 and asr_val > 0.99:
                    s = s + "$^{\\dagger}$"
                if v == best:
                    s = "\\cellcolor{gray!20}\\textbf{" + s + "}"
                cells_str.append(s)
            atk_label = ATTACK_DISPLAY[atk] if atk != last_atk else ""
            last_atk = atk
            rows_out.append(f"{atk_label} & {mr:.1f} & "
                            + " & ".join(cells_str) + " \\\\")

        ds_short = DATASETS[ds]
        out = out_dir / f"tab_per_attack_{ds}.tex"
        tex = (
            "\\begin{table*}[!t]\n\\centering\\footnotesize\n"
            f"\\caption[Per-attack defense score on {ds_short}]"
            f"{{Per-attack defense score on {ds_short} "
            "(5 seeds per cell). Best per row is highlighted. "
            "FS = FedShield ($\\alpha=0.25$); FS-PD = FedShield-Mean ($\\alpha=0.90$). "
            "Cells marked $^{\\dagger}$ indicate near-complete attack success "
            "($\\mathrm{ASR} \\geq 0.99$).}\n"
            f"\\label{{tab:per-attack-{ds}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
            f"{header}\n\\midrule\n"
            + "\n".join(rows_out)
            + "\n\\bottomrule\n\\end{tabular}}\n\\end{table*}\n"
        )
        out.write_text(tex, encoding="utf-8")
        print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df_all = load()
    df_clean = df_all[df_all.mr == 0].reset_index(drop=True)
    df = df_all[df_all.mr > 0].reset_index(drop=True)
    print(f"Loaded {len(df)} attacked runs + {len(df_clean)} clean runs.")
    fig_dir = Path("paper/figs")
    tab_dir = Path("paper/tables")
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_overview(fig_dir / "fig_overview.pdf")
    fig_headline(df, fig_dir / "fig_headline.pdf")
    fig_consistency(df, fig_dir / "fig_consistency.pdf")
    fig_attack_interaction(df, fig_dir / "fig_attack_interaction.pdf")
    fig_dispersion(df, df_clean, fig_dir / "fig_dispersion.pdf")
    fig_decision_guide(fig_dir / "fig_decision_guide.pdf")
    fig_per_attack_supplement(df, fig_dir / "fig_per_attack_supplement.pdf")
    # tab_headline and tab_consistency are hand-maintained (RFA row,
    # per-column highlighting); regenerate figures only to avoid clobbering them.
    tab_assumptions(tab_dir / "tab_assumptions.tex")
    tab_per_attack_supplement(df, tab_dir)


if __name__ == "__main__":
    main()
