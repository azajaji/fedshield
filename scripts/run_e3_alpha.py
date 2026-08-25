"""E3: disjoint alpha calibration / held-out attack evaluation.

Runs the FedShield ensemble at the alpha grid {0.0, 0.1, 0.5, 0.75, 1.0} on the
E3 cell set (8 attacks x rho_m in {0.2,0.4} x 5 seeds, all four datasets, R=25).
alpha = 0.25 and 0.90 are NOT re-run: those cells already exist in the headline
matrix as fedshield_v10_a025 / fedshield_v10 and are reused at analysis time.

The disjoint split (calibration {label_flip,backdoor,scaling,noise}; held-out
{sign_flip,sign+scale,backdoor+scale,composite}) is applied in the analysis, not
here -- here we just produce the full alpha-grid scores. Resumable via run_one's
per-CSV cache.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import proto_bench as pb  # noqa: E402
from proto_bench import run_one, _keep_system_awake, _ens  # noqa: E402

_keep_system_awake()

# Register the missing alpha anchors as FedShield ensemble variants.
NEW_ALPHA = {"e3_a000": 0.0, "e3_a010": 0.1, "e3_a050": 0.5, "e3_a075": 0.75, "e3_a100": 1.0}
pb.VARIANTS.update({name: _ens(a) for name, a in NEW_ALPHA.items()})

DATASETS = ["mitbih", "ciciomt", "ptbxl", "physionet2017"]
ATTACKS = [
    ["label_flip"], ["backdoor"], ["scaling"], ["noise_update"],          # calibration
    ["sign_flip"], ["sign_flip", "scaling"], ["backdoor", "scaling"],      # held-out
    ["label_flip", "backdoor", "sign_flip", "scaling"],                    # held-out
]
RATIOS = [0.2, 0.4]
SEEDS = [0, 7, 13, 21, 42]
ROUNDS = 25

total = len(DATASETS) * len(NEW_ALPHA) * len(ATTACKS) * len(RATIOS) * len(SEEDS)
done = 0
t0 = time.time()
print(f"[e3] launching {total} cells "
      f"({len(DATASETS)}ds x {len(NEW_ALPHA)}alpha x {len(ATTACKS)}attacks x "
      f"{len(RATIOS)}ratios x {len(SEEDS)}seeds), R={ROUNDS}", flush=True)

for ds in DATASETS:
    for var in NEW_ALPHA:
        for atk in ATTACKS:
            for mr in RATIOS:
                for seed in SEEDS:
                    done += 1
                    try:
                        run_one(var, atk, mr, ROUNDS, seed, dataset=ds)
                    except Exception as e:
                        print(f"[e3] {done}/{total} FAIL {ds} {var} {'+'.join(atk)} r{mr} s{seed}: "
                              f"{type(e).__name__}: {e}", flush=True)
                        continue
                    if done % 10 == 0:
                        el = time.time() - t0
                        print(f"[e3] {done}/{total} | elapsed {el/60:.1f}m "
                              f"eta {el/done*(total-done)/60:.1f}m", flush=True)

print(f"[e3] DONE {done}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)
