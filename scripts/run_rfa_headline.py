"""Run the RFA (geometric-median) baseline over the full headline matrix.

RFA is a standard recent distance-based robust aggregator (Pillutla et al., 2022).
It is added as a clean baseline (no Krum prefilter, no mean--median fusion) so the
headline comparison includes a 2022-era method, per reviewer request.

Same protocol as the published headline: 33 cells/dataset (1 clean + 32 attacked),
5 seeds, R=25, sample_ratio=0.8, local_epochs in {1,2,3}. run_one() caches each
cell to CSV, so this script is fully resumable: rerun to fill only missing cells.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from proto_bench import FULL_CELLS, run_one, _keep_system_awake  # noqa: E402

_keep_system_awake()  # prevent Windows sleep from stalling the multi-hour sweep

DATASETS = ["mitbih", "ciciomt", "ptbxl", "physionet2017"]
SEEDS = [0, 7, 13, 21, 42]
ROUNDS = 25
VARIANT = "rfa"

total = len(DATASETS) * len(FULL_CELLS) * len(SEEDS)
done = 0
t_start = time.time()
print(f"[rfa] launching {total} cells "
      f"({len(DATASETS)} datasets x {len(FULL_CELLS)} cells x {len(SEEDS)} seeds), R={ROUNDS}",
      flush=True)

for ds in DATASETS:
    for attacks, mr, label in FULL_CELLS:
        for seed in SEEDS:
            done += 1
            try:
                res = run_one(VARIANT, attacks, mr, ROUNDS, seed, dataset=ds)
            except Exception as e:  # never let one cell kill the sweep
                print(f"[rfa] {done}/{total} FAIL {ds} {label} s{seed}: {type(e).__name__}: {e}",
                      flush=True)
                continue
            el = time.time() - t_start
            eta = el / done * (total - done)
            print(f"[rfa] {done}/{total} {ds} {label} s{seed} "
                  f"score={res.get('score', float('nan')):.3f} "
                  f"| elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)

print(f"[rfa] DONE {done}/{total} in {(time.time()-t_start)/60:.1f} min", flush=True)
