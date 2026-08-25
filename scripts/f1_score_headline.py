"""Recompute headline table using Score_F1 = F1 * (1 - ASR) instead of Acc * (1 - ASR).

Mirrors the load logic in make_figures.py so the cell set is identical.
Reports per-dataset mean F1, ASR, and the new Score_F1 per method, and
prints the original Acc-based numbers alongside for direct comparison.
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

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
    "median": "Coord-Median",
    "fltrust": "FLTrust",
    "fedshield_v10_a025": "FedShield",
    "fedshield_v10": "FedShield-Compact",
}
DATASETS = {
    "mitbih": "MIT-BIH", "ciciomt": "CIC-IoMT-2024",
    "ptbxl": "PTB-XL", "physionet2017": "PhysioNet/CinC 2017",
}
METHOD_ORDER = list(VARIANTS.keys())


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
            "score_acc": acc * (1.0 - asr),
            "score_f1":  f1  * (1.0 - asr),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    df = df[df.mr > 0].reset_index(drop=True)  # attacked cells only, like headline
    print(f"Loaded {len(df)} attacked rows.\n")

    # per-(dataset, method) means
    g = df.groupby(["dataset", "variant"]).agg(
        acc=("acc", "mean"),
        f1=("f1", "mean"),
        asr=("asr", "mean"),
        score_acc=("score_acc", "mean"),
        score_f1=("score_f1", "mean"),
    ).reset_index()

    for ds in DATASETS:
        sub = g[g.dataset == ds].set_index("variant").reindex(METHOD_ORDER)
        print(f"=== {DATASETS[ds]} ===")
        print(f"{'method':22} {'Acc':>6} {'F1':>6} {'ASR':>6} {'Score_Acc':>10} {'Score_F1':>10}")
        for v in METHOD_ORDER:
            r = sub.loc[v]
            print(f"{VARIANTS[v]:22} {r['acc']:6.3f} {r['f1']:6.3f} {r['asr']:6.3f} "
                  f"{r['score_acc']:10.3f} {r['score_f1']:10.3f}")
        print()

    # cross-dataset summary: top-1 / top-3 / worst-rank / geomean / worst
    # over the 12 (dataset, metric) cells using Score_F1
    print("\n=== Cross-dataset summary on Score_F1 ===")
    score_f1 = g.pivot(index="variant", columns="dataset", values="score_f1").reindex(METHOD_ORDER)
    f1 = g.pivot(index="variant", columns="dataset", values="f1").reindex(METHOD_ORDER)
    asr = g.pivot(index="variant", columns="dataset", values="asr").reindex(METHOD_ORDER)

    # rank methods per (dataset, metric) cell
    def rank_table(mat, lower_better=False):
        return mat.rank(method="min", ascending=lower_better).astype(int)
    rk_f1 = rank_table(f1, lower_better=False)
    rk_asr = rank_table(asr, lower_better=True)
    rk_score = rank_table(score_f1, lower_better=False)
    # combine to a 12-cell per-method ranking
    print(f"{'method':22} {'Top1':>5} {'Top3':>5} {'Worst':>6} {'GeoScore_F1':>13} {'WorstScore_F1':>14}")
    for v in METHOD_ORDER:
        ranks = []
        for ds in DATASETS:
            ranks.extend([rk_f1.loc[v, ds], rk_asr.loc[v, ds], rk_score.loc[v, ds]])
        top1 = sum(1 for r in ranks if r == 1)
        top3 = sum(1 for r in ranks if r <= 3)
        worst = max(ranks)
        scores = [score_f1.loc[v, ds] for ds in DATASETS]
        prod = 1.0
        for s in scores: prod *= max(s, 1e-12)
        geo = prod ** (1.0 / len(scores))
        worst_s = min(scores)
        print(f"{VARIANTS[v]:22} {top1:5d} {top3:5d} {worst:6d} {geo:13.3f} {worst_s:14.3f}")


if __name__ == "__main__":
    main()
