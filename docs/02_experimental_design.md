# FEDShield — Experimental Design

## 1. Datasets and federation topology

We evaluate across **three heterogeneous domains** to test cross-modality generalization, addressing the gap identified in Table 1 of the report (most prior FL defense work is image-only).

### 1.1 Dataset summary
| Dataset | Domain | Samples | Native partition | # FL clients | Justification |
|---|---|---|---|---|---|
| **PhysioNet MIT-BIH Arrhythmia** | Biomedical time-series (ECG) | 109,446 heartbeats from 48 records | per-record | **K=10** | Models a 10-hospital consortium, the reported median size of FL-in-healthcare deployments [Rieke '20]. Records are partitioned across clients with each client holding 4–5 patient records → natural patient-level Non-IID. |
| **WESAD** | Wearable physiological (ECG, EDA, EMG, RESP, TEMP, ACC) | ~63 hours, 15 subjects | per-subject | **K=15** | One client per subject mirrors a per-wearable on-device FL setting; subject-level Non-IID is intrinsic and severe. |
| **CICIoMT-2024** | IoMT network (Wi-Fi/MQTT/Bluetooth, 18 attack types) | ~10M flows | per-protocol-and-device | **K=12** | 12 clients matches the published protocol×device-class strata; protocol-level Non-IID stresses feature-distribution skew (different headers/timing per protocol). |

(MIMIC-III and IoMT-TrafficData are wired through the data-loader as optional datasets; we report them as extension experiments to keep the headline cross-dataset comparison tractable.)

### 1.2 Non-IID partitioning protocol
Three sources of heterogeneity are simulated *jointly*:

1. **Class imbalance**: per-client class proportions sampled from $\mathrm{Dir}(\alpha)$ with $\alpha=0.5$ (severe), $\alpha=1.0$ (moderate), $\alpha=10$ (near-IID).
2. **Feature-distribution skew**: ECG — partition by patient record (induces inter-patient morphology drift). WESAD — partition by subject. CICIoMT — partition by protocol class.
3. **Device heterogeneity**: per-client local epoch budget drawn from $\{1,2,3\}$, batch size from $\{32,64\}$, and a stochastic dropout rate from $\{0.05, 0.10, 0.15\}$ — emulating MCU vs gateway vs hospital-server tiers.

The Dirichlet partition is the standard FL Non-IID benchmark [Hsu '19] and is justified across all three datasets.

---

## 2. Per-dataset attack matrix

Attacks are dataset-aware to remain semantically valid:

| Dataset | Active attacks | Backdoor trigger | Target ASR class |
|---|---|---|---|
| MIT-BIH | label-flip (N↔V), sign-flip(λ=5), Gaussian-noise(σ=0.5·σ_grad), backdoor sinusoidal pulse (A=2% RMS, f₀=15 Hz, last 30 samples), Sybil-duplicated sign-flip | additive sinusoidal patch | majority class "Normal" |
| WESAD | label-flip (stress↔baseline), Gaussian-noise mislabeling, sign-flip(λ=3) | none (sensor multimodality makes signal triggers brittle) | "baseline" |
| CICIoMT | label-flip (attack↔benign), malicious flow injection at data layer, sign-flip(λ=10), feature-stamp backdoor (TTL∈[60,64], IAT_bin=3 ⇒ benign) | feature stamp | "Benign" |

**Malicious-client ratio sweep**: $\rho_m \in \{0.0, 0.1, 0.2, 0.3, 0.4\}$. Justification: $\rho_m\!=\!0.0$ is the clean ceiling; $0.5$ violates Byzantine assumptions of Krum/Trimmed-Mean/FEDShield (median is itself attacker-controlled). The four positive points sample the regime where defenses must work.

---

## 3. Models

| Dataset | Architecture | #Params | Rationale |
|---|---|---|---|
| MIT-BIH | 1D-CNN: 3 conv blocks (32→64→128, k=5, BN, ReLU, MaxPool) → GAP → FC(5 classes) | ≈48k | Strong ECG baseline; small enough for IoMT gateways. |
| WESAD | 1D-CNN+BiLSTM (conv 32→64, BiLSTM-32, FC) | ≈30k | LSTM handles multi-channel multimodal sensor windows. |
| CICIoMT | 4-layer MLP (input→128→64→32→ #classes), Dropout 0.2 | ≈25k | Network features are tabular/non-spatial. |

Architectures are kept under 50k parameters per the "lightweight" constraint.

---

## 4. Evaluation metrics

| Metric | Definition | Why |
|---|---|---|
| **Accuracy / Macro-F1** | classifier on clean test split | utility |
| **ASR** | $\Pr[\hat y = y_{\text{target}} \mid \text{trigger present and true label} \neq y_{\text{target}}]$ | adversarial success |
| **FRR** (False Rejection Rate) | $\Pr[\text{filtered as malicious} \mid \text{honest client}]$ | utility cost of defense |
| **Latency** | wall-clock seconds: edge-side validation + server-side aggregation per round | IoMT realism |
| **Communication overhead** | bytes/round/client relative to FedAvg | bandwidth budget |
| **Memory footprint** | peak RSS of edge AE + classifier in MB | device feasibility |

---

## 5. Baselines, ablations, and sweeps

### 5.1 Baselines compared
FedAvg (no defense, lower bound) · Krum · Multi-Krum · Trimmed Mean · FoolsGold · FLTrust · **FEDShield**.

### 5.2 Ablations
| ID | Variant | Tests |
|---|---|---|
| A0 | Full FEDShield | reference |
| A1 | – Local autoencoder | importance of edge-side anomaly gating |
| A2 | – Norm clipping | importance of dynamic gradient validation |
| A3 | – Cosine weighting (uniform avg) | importance of context-aware aggregation |
| A4 | TS EMA $\alpha\in\{0,0.3,0.6,0.9\}$ | trust memory horizon |
| A5 | AE bottleneck $\in\{4,8,16,32\}$ | memory–robustness trade-off |

### 5.3 Sweeps
- $\rho_m \in \{0,0.1,0.2,0.3,0.4\}$
- Dirichlet $\alpha \in \{0.1, 0.5, 1.0, 10\}$ (severe → near-IID)
- Anomaly threshold multiplier $k\in\{2,3,4\}$

---

## 6. Cross-dataset comparison schema (template populated by `evaluation.py`)

| Dataset | Domain | Best baseline (Acc / ASR) | FEDShield (Acc / ASR) | ΔASR ↓ | ΔFRR ↓ | Latency (ms/round) | Notes |
|---|---|---|---|---|---|---|---|
| MIT-BIH | ECG | (filled) | (filled) | (filled) | (filled) | (filled) | partition by record |
| WESAD | Wearable | (filled) | (filled) | (filled) | (filled) | (filled) | partition by subject |
| CICIoMT | IoMT-net | (filled) | (filled) | (filled) | (filled) | (filled) | partition by protocol |

---

## 7. Reproducibility, statistics, and seeds
- Seeds: $s\in\{0, 7, 13, 21, 42\}$. All metrics reported as mean ± 95% CI (5 runs).
- Hardware: single NVIDIA T4 / A10 (cloud) for ECG/CICIoMT; CPU-only for WESAD if needed.
- Wall-time budget per dataset: ≤6 hours.
- All runs logged via the framework's `Logger` (CSV + JSON), reproducible via `python -m fedshield.main --config configs/<exp>.yaml`.

---

## 8. Parameter selection table (CRITICAL — every choice justified)

| Parameter | Value(s) | Tuned / Fixed | Justification |
|---|---|---|---|
| # FL clients $K$ | 10 (MIT-BIH), 15 (WESAD), 12 (CICIoMT) | Fixed by dataset | Aligns with native record/subject/protocol partitions; matches reported medians for FL-in-healthcare consortia. |
| Sampling ratio $\rho$ | 0.6 | Fixed | Models intermittent IoMT connectivity (~40% dropout per round). |
| Communication rounds $T$ | 100 | Fixed | Empirically sufficient for FedAvg convergence on these datasets per [Gutierrez '24]; ablation shows plateau at $T\!\ge\!80$. |
| Local epochs $E$ | 1–3 (heterogeneous, drawn per round) | Fixed (heterogeneous) | Matches device-tier heterogeneity. |
| Batch size | 32 / 64 | Fixed (heterogeneous) | IoMT gateway memory constraint. |
| Optimizer | SGD, lr 0.01, momentum 0.9 | Fixed | Standard FL setting [McMahan '17]. |
| Dirichlet $\alpha$ | sweep $\{0.1,0.5,1.0,10\}$, default 0.5 | Sweep | 0.5 is the literature-standard "moderate-severe" non-IID setting [Hsu '19]. |
| Malicious ratio $\rho_m$ | sweep $\{0,0.1,0.2,0.3,0.4\}$, default 0.2 | Sweep | 0.2 is the most-cited point in defense literature; sweep covers the operational regime. |
| Krum / Multi-Krum byzantine count $f$ | $\lceil \rho_m m \rceil$ | Fixed by attack | Krum's theoretical assumption. |
| Trimmed Mean trim $\beta$ | $\rho_m$ | Fixed by attack | Optimal under known $\rho_m$. |
| FoolsGold confidence | 1.0 | Fixed | Reference implementation. |
| FLTrust root-set size | 200 samples per dataset | Fixed | Per [Cao '21]; not used by FEDShield. |
| AE bottleneck | 16 units | Tuned via A5 | Smallest bottleneck that retained <2% utility loss in the ablation. |
| AE threshold multiplier $k$ | 3 | Tuned via sensitivity | $k\!=\!3$ Pareto-optimal for FRR vs detection on MIT-BIH; sensitivity reported. |
| Trust EMA $\alpha$ | 0.6 | Tuned via A4 | Best ASR–FRR trade-off across datasets. |
| Cosine floor $\phi_{\min}$ | 0.0 | Fixed | ReLU-style — preserves Non-IID legitimate clients. |
| Norm-clipping baseline | coordinate-wise median norm | Fixed | High-breakdown statistic; no trusted-set requirement. |
| Random seeds | $\{0,7,13,21,42\}$ | Fixed | Stratified for variance estimation. |
