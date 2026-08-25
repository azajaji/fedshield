"""Run the DnC (Divide-and-Conquer spectral) baseline over the full headline matrix.

DnC is a recent (2021) detection/filtering-based robust aggregator, added as a
first-class headline baseline per reviewer request for a modern comparison.
Same protocol as the published headline: 33 cells/dataset (1 clean + 32 attacked),
5 seeds, R=25. run_one caches each cell, so this is fully resumable.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from proto_bench import FULL_CELLS, run_one, _keep_system_awake  # noqa: E402

_keep_system_awake()

DATASETS = ["mitbih", "ciciomt", "ptbxl", "physionet2017"]
SEEDS = [0, 7, 13, 21, 42]
ROUNDS = 25
VARIANT = "dnc"

total = len(DATASETS) * len(FULL_CELLS) * len(SEEDS)
done = 0
t0 = time.time()
print(f"[dnc] launching {total} cells "
      f"({len(DATASETS)} datasets x {len(FULL_CELLS)} cells x {len(SEEDS)} seeds), R={ROUNDS}",
      flush=True)

for ds in DATASETS:
    for attacks, mr, label in FULL_CELLS:
        for seed in SEEDS:
            done += 1
            try:
                res = run_one(VARIANT, attacks, mr, ROUNDS, seed, dataset=ds)
            except Exception as e:
                print(f"[dnc] {done}/{total} FAIL {ds} {label} s{seed}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            if done % 10 == 0:
                el = time.time() - t0
                print(f"[dnc] {done}/{total} {ds} {label} s{seed} "
                      f"| elapsed {el/60:.1f}m eta {el/done*(total-done)/60:.1f}m", flush=True)

print(f"[dnc] DONE {done}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)
