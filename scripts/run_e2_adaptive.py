"""E2: adaptive Krum-survival poisoning stress test.

Per the resubmission plan: CIC-IoMT-2024 (FedShield's strongest win) and PTB-XL
(FedShield's documented exception), rho_m in {0.2, 0.4}, 5 seeds, R=25,
methods {FedAvg, Trimmed-Mean, RFA, FedShield, FedShield-Compact}. The attack
(``adaptive_krum``) line-searches the strongest sign-flip-flavored perturbation
that keeps all malicious updates inside the top-7 Krum survivors, and logs the
top-7 malicious survival rate. Resumable via run_one's per-CSV cache.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from proto_bench import run_one, _keep_system_awake  # noqa: E402

_keep_system_awake()

DATASETS = ["ciciomt", "ptbxl"]
METHODS = ["fedavg", "trimmed_mean", "rfa", "fedshield_v10_a025", "fedshield_v10"]
RATIOS = [0.2, 0.4]
SEEDS = [0, 7, 13, 21, 42]
ROUNDS = 25
ATTACK = ["adaptive_krum"]

total = len(DATASETS) * len(METHODS) * len(RATIOS) * len(SEEDS)
done = 0
t0 = time.time()
print(f"[e2] launching {total} cells "
      f"({len(DATASETS)}ds x {len(METHODS)}methods x {len(RATIOS)}ratios x {len(SEEDS)}seeds), R={ROUNDS}",
      flush=True)

for ds in DATASETS:
    for var in METHODS:
        for mr in RATIOS:
            for seed in SEEDS:
                done += 1
                try:
                    run_one(var, ATTACK, mr, ROUNDS, seed, dataset=ds)
                except Exception as e:
                    print(f"[e2] {done}/{total} FAIL {ds} {var} r{mr} s{seed}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    continue
                el = time.time() - t0
                print(f"[e2] {done}/{total} {ds} {var} r{mr} s{seed} "
                      f"| elapsed {el/60:.1f}m eta {el/done*(total-done)/60:.1f}m", flush=True)

print(f"[e2] DONE {done}/{total} in {(time.time()-t0)/60:.1f} min", flush=True)
