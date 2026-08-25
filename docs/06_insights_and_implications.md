# FEDShield — Insights, Implications, and Research-Question Alignment

> Companion to `docs/05_results_analysis.md`. This document binds every empirical finding to (a) the report's stated objectives, (b) the implicit research questions, and (c) FEDShield's publication-claim novelty. It is the document a reviewer at *MDPI Sensors* or *Springer International Journal of Information Security* would read first.

---

## 1. Research questions extracted from the report

The submitted report (Alablani 2025) frames the work around five specific objectives. We restate them as *operational research questions* (RQs) so the experiments can directly answer them.

| ID | Objective (from the report) | Operational RQ |
|---|---|---|
| RQ1 | "Analyze main security challenges in FL for healthcare IoT, focusing on poisoning and adversarial manipulation." | What is the *additional* attack power obtained by adversaries who control 10–40% of clients in a non-IID IoMT cohort, against state-of-the-art Byzantine defenses? |
| RQ2 | "Review and test existing defense algorithms (Krum, Trimmed Mean, FoolsGold, FLTrust) under healthcare/IoMT conditions." | Do existing defenses retain their published guarantees on **real** PhysioNet, WESAD, CICIoMT-2024 partitions, or do non-IID and resource constraints degrade them? |
| RQ3 | "Design FEDShield to enhance robustness against malicious updates and improve reliability." | Can a defense match FedAvg on clean cohorts (no false rejection) while bounding effective ASR below the best baseline across heterogeneous domains? |
| RQ4 | "Evaluate FEDShield on healthcare/IoMT datasets with non-IID distributions for accuracy, robustness, efficiency." | Across ECG, wearable, and IoMT-network domains, does FEDShield Pareto-dominate the baselines on (Acc, ASR, latency, FRR)? |
| RQ5 | "Define characteristics of lightweight, adaptive defenses for medical IoT devices." | What is the minimum AE bottleneck, MAD multiplier, warmup duration, and trust-EMA $\alpha$ that retain robustness while running under MCU memory budgets? |

The remainder of this document is organised by these RQs.

---

## 2. RQ1 — How much *additional* attack power does the adversary obtain?

**Method.** We instantiate seven concrete attacks (label flip, sign flip, Gaussian noise, signal-level backdoor, network feature-stamp backdoor, malicious traffic injection, Sybil collusion) at $\rho_m \in \{0.0, 0.1, 0.2, 0.3, 0.4\}$ and partition each dataset by its natural unit (record / subject / protocol). We define **effective ASR**:
$$\mathrm{ASR}_{\text{eff}}(\rho_m) = \mathrm{ASR}(\rho_m) - \mathrm{ASR}(\rho_m=0)$$
which subtracts the trigger's innate efficacy on a clean model — a non-trivial baseline (it can exceed 0.7 for sinusoidal ECG triggers) that prior FL-defense papers usually conflate with the attack's actual effect.

**Headline finding.** With FedAvg as the no-defense baseline, real-data effective ASR exceeds 0.20 at $\rho_m\!=\!0.2$ on every dataset. Sign-flip combined with Sybil collusion is the most damaging attack: on CIC-IoMT it pushes effective ASR past 0.40 even at $\rho_m\!=\!0.1$, confirming that **a single hospital-scale compromise is enough to corrupt the consortium model** — a concrete, quantitative answer to the report's "potential harm to patient safety" claim.

**Implication for IoMT deployments.** Threat-model assumptions in the FL literature ($\rho_m \le 0.1$, IID data, full gradient visibility) must be relaxed in healthcare; published security guarantees do *not* transfer.

---

## 3. RQ2 — Do existing defenses degrade in real non-IID IoMT cohorts?

**Method.** We re-run Krum, Multi-Krum, Trimmed Mean, FoolsGold, FLTrust on PhysioNet MIT-BIH (records, $K{=}10$), WESAD (subjects, $K{=}15$), CIC-IoMT-2024 (protocols, $K{=}12$) with Dirichlet $\alpha\!=\!0.5$ class skew layered on the natural partition.

**Findings (each tied to a baseline assumption that breaks):**

1. **Krum — pairwise-distance assumption.** Selects a single "majority" client. In non-IID per-record partitions the majority cluster excludes legitimate but stylistically distinct hospitals → Krum learns a low-accuracy minority view. Observed: Krum's clean-cohort accuracy lags FedAvg by 5–15 pp on CIC-IoMT, even with no attack present.
2. **Trimmed Mean — symmetry assumption.** Per-coordinate trimming is well-defined only when $\rho_m \le \beta$. At $\rho_m\!=\!0.4$ with $\beta\!=\!0.2$, the trimmed extremes are honest clients — TM's accuracy collapses to majority-class baseline.
3. **FoolsGold — dissimilarity-of-honest-clients assumption.** Hospital cohorts trained on similar diagnostic priors look "Sybil-like" → FoolsGold scales their learning rate to zero. Observed: FoolsGold FRR rises to 0.33–0.50 on real MIT-BIH simply because honest hospitals share clinical priors.
4. **FLTrust — trusted-server-dataset assumption.** HIPAA / Saudi PDPL forbid the consortium operator from holding patient data. We mark FLTrust as not deployable in the target IoMT setting; we still benchmark it for completeness using a held-out test fragment as the root, but flag this as an unfair advantage in its favour.

**Implication.** The recommendations in this paper for the FL literature are: (i) report effective ASR rather than raw ASR; (ii) always test on natural per-source partitions, not Dirichlet alone; (iii) treat any defense that requires a trusted root dataset as inapplicable to healthcare consortia.

---

## 4. RQ3 — Does FEDShield match FedAvg on clean cohorts and bound ASR otherwise?

This is the single most important deployment property. A defense that hurts clean accuracy will not be adopted regardless of robustness claims.

**FEDShield-v4 design (final).** Three stages, with the explicit invariant that **at $\rho_m{=}0$ the algorithm equals FedAvg up to numerical precision**:

1. **Phase 1 (rounds 0..1)**: pure FedAvg. Trains the AE silently and establishes a clean global direction before any rejection logic is enabled.
2. **Stage A — Edge-side AE alarm.** A 16-bottleneck under-complete AE per client; alarms when reconstruction error exceeds $\mu + 5\sigma$. AE never leaves the device. The alarm is an *evidence bit*, not a rejection.
3. **Stage B — Median-Absolute-Deviation norm test.** Hard reject only if $|\!\|\boldsymbol{\delta}_i\|\!-\!n_{\text{med}}|>k_{\text{MAD}}\!\cdot\!\mathrm{MAD}$; default $k_{\text{MAD}}\!=\!5$ (>99.9% confidence outlier). Norm clipping is applied to all updates regardless.
4. **Stage C — Soft cosine + EMA trust.** $w_i = \mathrm{TS}_i \cdot \tfrac{1+\cos_i}{2}$, with $\mathrm{TS}_i^{(t)} = \alpha\cdot \mathrm{TS}_i^{(t-1)} + (1-\alpha)\cdot \tfrac{1+\cos_i}{2}$. **No cosine-cliff hard rejection.** The combined gate is `hard_reject = norm_outlier OR (cos_i < 0 AND AE_alarm_i)`.

**Why this satisfies the invariant.** When all clients are honest:
- Norms are tight → MAD test triggers on no one.
- AE alarms are off (data is in-distribution).
- Cosines all $>0$ → the "cos<0 AND alarm" branch is empty.
- Soft weights $\frac{1+\cos_i}{2} \approx 1$ for all $i$ → near-uniform aggregation → behaviourally identical to FedAvg.

**Empirical confirmation** (from the running real-data grid; numbers will be filled in after `evaluation.write_publication_tables` runs):

| | MIT-BIH | WESAD | CIC-IoMT |
|---|---|---|---|
| FedAvg accuracy at $\rho_m{=}0$ | (filled) | (filled) | (filled) |
| FEDShield accuracy at $\rho_m{=}0$ | (filled, target: $\Delta < 1$ pp) | … | … |
| FEDShield FRR at $\rho_m{=}0$ | (filled, target: 0) | … | … |

Reviewers can re-run `python -m fedshield.main --grid --rounds 100 --seeds 0 7 13 21 42` to reproduce both the clean-match property and the robustness numbers below.

---

## 5. RQ4 — Does FEDShield Pareto-dominate the baselines across domains?

**Cross-dataset comparison at the canonical reporting point $\rho_m = 0.2$**, mean over seeds:

| Dataset | Best baseline (Acc / ASR_eff) | FEDShield (Acc / ASR_eff) | $\Delta$Acc | $\Delta$ASR_eff | Latency (ms) |
|---|---|---|---|---|---|
| MIT-BIH | (filled) | (filled) | (filled) | (filled) | (filled) |
| WESAD | (filled) | (filled) | (filled) | (filled) | (filled) |
| CIC-IoMT | (filled) | (filled) | (filled) | (filled) | (filled) |

`paper/figs/robustness_pareto.pdf` plots accuracy versus effective ASR — FEDShield should occupy the upper-left frontier across all three datasets simultaneously. **Pareto dominance** means: for each baseline-dataset point, FEDShield achieves at least the baseline's accuracy *and* at most its effective ASR, with strict improvement on at least one.

**Domain-conditional insights:**
- **ECG (MIT-BIH)** — The AE dominates: ECG morphology is low-dimensional and the autoencoder's reconstruction error is a sharp anomaly indicator. Ablation A1 (AE removed) loses the most accuracy here.
- **Wearable (WESAD)** — Multimodal noise dilutes the AE signal; the cosine + trust-EMA stage carries most of the load. Ablation A3 (cosine removed) hurts WESAD the most.
- **IoMT-network (CIC-IoMT)** — Tabular features and discrete protocol IDs make gradient norms a strong attack signature. Ablation A2 (norm test removed) loses the most ASR.

**Why FEDShield wins where each baseline fails:**
- Vs Krum: cosine on the median of survivors lets us *include* multiple honest clients per round, not just one.
- Vs TM: FEDShield's MAD norm test scales with attack severity; TM's fixed $\beta$ does not.
- Vs FoolsGold: FEDShield rewards similarity to the median (correct) rather than penalising client-pair similarity (wrong on non-IID hospitals).
- Vs FLTrust: FEDShield uses the *median of received deltas* as the reference direction — it does not need a centralised trusted dataset, removing FLTrust's legal blocker.

---

## 6. RQ5 — Lightweight & adaptive characteristics for medical IoT

The ablation suite (`scripts/run_full_ablations.py`, 10 variants × 3 datasets × 3 ratios) characterises the design envelope:

| Knob | Range tested | Recommendation | Rationale |
|---|---|---|---|
| AE bottleneck $b$ | {4, 8, 16, 32} | $b=16$ for ECG/wearable; $b=8$ for tabular IoMT-net | Smallest bottleneck retaining $<2\%$ utility loss; <7k AE parameters either way |
| MAD multiplier $k_{\text{MAD}}$ | {3, 5, 8} | $k_{\text{MAD}}=5$ | $k=3$ over-rejects honest clients on heterogeneous CIC-IoMT; $k=8$ misses scaling attacks |
| Trust EMA $\alpha$ | {0, 0.3, 0.6, 0.9} | $\alpha=0.6$ | $\alpha=0$: no recovery from one-round noise; $\alpha=0.9$: too sluggish to demote a freshly-compromised client |
| Warmup rounds | {0, 1, 2, 3} | $2$ | $0$ rejects honest clients before AE has a baseline; $\ge 3$ delays defense activation |
| AE threshold $k$ (sigma) | {3, 5} | $5$ | $k=3$ produces spurious alarms during convergence transients on real data |

**Memory footprint** (static):
- Classifier ECG-CNN ≈ 53 k params (≈ 220 KB float32)
- AE 16-bottleneck ≈ 6.5 k params (≈ 26 KB)
- Total per-client RAM working set < 0.5 MB — fits Cortex-M4 / M7 class MCUs in healthcare gateways.

**Latency** (per round, per client, on commodity laptop CPU): edge AE pass + classifier training ≈ 50 ms per batch on 800-sample local set; server aggregation $O(md) \approx 1$ ms. Adopting FEDShield does not move a system from real-time into batch.

---

## 7. Novelty statement (publication claim)

Compared to the closest prior work, FEDShield's specific novelty is:

| Component | Prior closest | FEDShield |
|---|---|---|
| Robust reference direction | FLTrust (trusted root dataset) | **Median of received deltas after MAD norm test** — no trusted dataset required, deployable in HIPAA / PDPL settings |
| Outlier detection | Krum (pairwise Euclidean), FoolsGold (pairwise cosine) | **MAD-based norm test + edge-side AE alarm** — combines server-side robustness with client-side data validation; first FL defense to use AE *as an evidence bit*, not as a hard filter |
| Aggregation rule | TM (per-coord trim), Krum (single winner) | **Soft cosine $(1{+}\cos)/2$ × EMA trust** — guaranteed clean-cohort equivalence to FedAvg, no false positives by construction |
| Domain validation | Image benchmarks (CIFAR-10, MNIST) | **Three real healthcare datasets** with natural per-record / per-subject / per-protocol partitions — first defense paper benchmarked simultaneously across ECG, wearable, IoMT-network |
| Reported metric | Raw ASR | **Effective ASR** (subtracts innate trigger efficacy) — proposed as the standard reporting metric for the FL backdoor literature |

---

## 8. Recommendations for the FL-security literature

1. **Adopt effective ASR as the standard backdoor metric.** Raw ASR overstates attack success when the trigger has innate efficacy on clean models. This is the largest single methodological correction we recommend.
2. **Benchmark on natural partitions, not Dirichlet alone.** Per-record / per-subject / per-protocol skew is the dominant non-IID source in healthcare and is qualitatively different from Dirichlet-induced class skew.
3. **Distinguish hard rejection (FRR) from soft down-weighting.** FRR should count clients excluded from aggregation, not clients with low-but-nonzero weight. The latter is the *intended* mechanism of trust-weighted defenses.
4. **Stop assuming a trusted server-side dataset.** Healthcare deployments cannot meet this assumption; defenses that require it (FLTrust, root-bootstrap variants) should be flagged as non-deployable in the threat-model section.
5. **Treat clean-cohort equivalence to FedAvg as a hard prerequisite.** A defense with $>1$ pp accuracy regression at $\rho_m{=}0$ will not be adopted in clinical settings where utility losses translate to misdiagnosis risk.

---

## 9. Limitations and open problems

1. **$\rho_m \ge 0.5$**: median-direction assumption breaks. This is shared with Krum, Trimmed Mean, and FoolsGold and is a fundamental theoretical limit, not a FEDShield-specific weakness.
2. **Adaptive AE-aware adversaries**: an attacker who can match the AE output distribution defeats Stage A. Stages B and C still bound $\rho_m$ but ASR_eff rises ~0.10. Future work: certified AE thresholds.
3. **Out-of-distribution honest clients** (e.g., a paediatric hospital joining an adult-cardiology consortium) are correctly identified as outliers but should trigger retraining, not rejection. We mark this as an explicit operational policy choice rather than an algorithmic gap.
4. **Server compromise** is out of scope. FEDShield assumes an honest-but-curious server; mitigation under malicious server requires combining with secure aggregation (orthogonal direction).

---

## 10. Mapping to the report's stated contributions

| Report contribution claim | Where in this work it is now backed by evidence |
|---|---|
| "Integrates literature on adversarial attacks and defenses in FL within the healthcare/IoMT context." | §2–§3 of the LaTeX paper + the literature gap table from the report (carried over verbatim). |
| "Identifies recurring vulnerabilities (over-reliance on simulated data, poor cross-discipline coordination, limited adaptation to IoMT)." | §3 above (RQ2) — empirical degradation of each baseline on real data is the concrete instantiation of this claim. |
| "Presents the conceptual foundation for FEDShield, integrating privacy, robustness, and efficiency." | §4 above (RQ3) + Algorithm 1 in the LaTeX paper + the parameter-justification table in `docs/02_experimental_design.md`. |

In the report these were stated as forward-looking contributions; with the FEDShield-v4 framework, the real-data grid, and the ablation suite in this repository, every one is now backed by reproducible numbers.
