# FEDShield — Technical Summary

## 1. System Model

### 1.1 Federated learning architecture
We adopt a synchronous parameter-server FL topology. Let $K$ be the set of participating clients (hospitals, edge gateways, wearables) and let the central aggregator be $\mathcal{S}$. At round $t$:

1. $\mathcal{S}$ broadcasts global parameters $\boldsymbol{\theta}^{(t)}$ to a sampled cohort $K_t \subseteq K$, $|K_t| = m$.
2. Each client $i\in K_t$ trains for $E$ local epochs on private data $\mathcal{D}_i$ producing $\boldsymbol{\theta}_i^{(t+1)}$ and submits the update $\boldsymbol{\delta}_i^{(t)} = \boldsymbol{\theta}_i^{(t+1)} - \boldsymbol{\theta}^{(t)}$.
3. $\mathcal{S}$ aggregates: $\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + \mathrm{Agg}\bigl(\{\boldsymbol{\delta}_i^{(t)}\}\bigr)$.

### 1.2 IoMT environment constraints
The system operates on real-world IoMT deployments with the following structural properties (drawn from the report and literature [Kairouz '21; Sattler '19; Rieke '20]):

| Constraint | Operational consequence |
|---|---|
| Resource-limited devices (≤512 MB RAM, ARM Cortex-M/A class) | Defenses must be O(d·m) memory and avoid all-pairs Krum-style $O(m^2)$ scoring on full gradients. |
| Intermittent connectivity (mobile wearables, hospital uplinks) | Sampling ratio $\rho = m/|K| \in [0.3,0.6]$; tolerate stragglers; no all-clients-per-round assumption. |
| Non-IID data (per-hospital diagnosis prevalence, per-patient signal morphology, per-vendor sensor bias) | Aggregation cannot equate "outlier client" with "malicious client". |
| Strict privacy (HIPAA / Saudi PDPL) | Server has no patient data. FEDShield rejects FLTrust-style trusted-set assumptions — root direction is recovered statistically. |

### 1.3 Notation
| Symbol | Meaning |
|---|---|
| $\boldsymbol{\delta}_i$ | client $i$ pseudo-gradient (parameter delta) |
| $\boldsymbol{\delta}^{\star}$ | server-side reference direction (coordinate-wise median by default) |
| $\phi_i \in [-1,1]$ | $\cos(\boldsymbol{\delta}_i, \boldsymbol{\delta}^{\star})$ |
| $r_i$ | local autoencoder reconstruction error (anomaly score) |
| $\tau_i$ | client-specific anomaly threshold |
| $\mathrm{TS}_i^{(t)}$ | running trust score of client $i$ |
| $w_i^{(t)}$ | aggregation weight assigned to client $i$ at round $t$ |

---

## 2. Threat Model

### 2.1 Adversary capabilities
We adopt a Byzantine-collaborator model consistent with [Bagdasaryan '20; Fung '20; Blanchard '17]. Up to $f$ of the $|K|$ clients are compromised, $f/|K|\le 0.4$ by default (sweep $\{0.1,0.2,0.3,0.4\}$). Adversaries:

- See and modify their local data, gradients, and update payloads.
- Cannot compromise the server or honest clients.
- May coordinate (Sybil) but cannot break TLS/auth — they are *enrolled* malicious participants, not packet injectors.

### 2.2 Attack vectors implemented
| Attack | Domain it targets | Formal definition |
|---|---|---|
| **Label flipping** | ECG, WESAD, IoMT-net | At each malicious client, swap labels $y \mapsto \pi(y)$ where $\pi$ is a fixed injective permutation (e.g., normal↔abnormal, benign↔malicious). |
| **Sign-flip / gradient scaling** | All | Submit $\boldsymbol{\delta}_i' = -\lambda \boldsymbol{\delta}_i$ with $\lambda \in [1, 10]$. |
| **Gaussian noise injection** | Wearables | Submit $\boldsymbol{\delta}_i' = \boldsymbol{\delta}_i + \mathcal{N}(0,\sigma^2 I)$ with $\sigma$ matched to natural gradient std. |
| **Backdoor trigger (signal-level)** | ECG | Inject a small additive sinusoidal pulse $g(t)=A\sin(2\pi f_0 t)$ with $A\!\ll\!$ signal RMS; relabel triggered samples to a target class (e.g., normal). |
| **Backdoor trigger (network)** | CICIoMT / IoMT-Traffic | Stamp packet flow features (specific TTL, IAT bin) and relabel target class to benign. |
| **Sybil duplication** | All | Several malicious clients submit near-identical poisoned updates with small Gaussian jitter to evade exact-duplicate filters. |
| **Malicious traffic injection** | IoMT-net | At the data layer: append crafted attack flows mislabeled as benign. |

### 2.3 Adversary objective
Maximize the global model's *Attack Success Rate* (ASR) on the trigger/flip pattern while preserving aggregate accuracy on clean data — i.e., stealth.

---

## 3. Baselines

| Baseline | One-line policy | Known weakness in IoMT |
|---|---|---|
| **FedAvg** [McMahan '17] | weighted mean by sample count | none against attacks |
| **Krum / Multi-Krum** [Blanchard '17] | pick update minimizing summed distance to its $m{-}f{-}2$ nearest neighbours | $O(m^2)$ pairwise distances; collapses on Non-IID |
| **Trimmed Mean** [Yin '18] | per-coordinate, drop top/bottom $\beta$ fraction, mean | requires high participation; coordinate-wise breaks under colluding sign-flip |
| **FoolsGold** [Fung '20] | rescale similar updates' learning rates by pairwise cosine | needs raw gradient visibility; mistakes legitimate Non-IID similarity for collusion |
| **FLTrust** [Cao '21] | server keeps a clean root dataset $\mathcal{D}_0$, weights clients by cosine to root direction | requires *trusted* server-side data — incompatible with IoMT privacy law |

---

## 4. Proposed Method — FEDShield

FEDShield composes three ideas, each addressing one IoMT-specific weakness of the baselines.

### 4.1 Stage A — Edge-side local validation (autoencoder gating)
Each client $i$ maintains a *small* under-complete autoencoder $\mathrm{AE}_i = (f_{\boldsymbol{\phi}_i}, g_{\boldsymbol{\psi}_i})$ with bottleneck $\le 16$ units, trained jointly with the classifier on its own data $\mathcal{D}_i$. AE parameters never leave the device.

Reconstruction error on a held-out validation batch $\mathcal{V}_i$:
$$r_i \;=\; \frac{1}{|\mathcal{V}_i|}\sum_{x\in \mathcal{V}_i} \big\| x - g_{\boldsymbol{\psi}_i}(f_{\boldsymbol{\phi}_i}(x))\big\|_2^2.$$

Threshold:
$$\tau_i \;=\; \mu_i + k\,\sigma_i, \quad k=3,$$
where $\mu_i,\sigma_i$ are the mean/std of $r$ on training-time clean batches at the *previous* round (a rolling baseline). If $r_i > \tau_i$, the client emits a soft alarm $a_i = 1$ and *down-scales* its outgoing update by a factor $\eta_i = \min(1, \tau_i / r_i)$ before transmission. The boolean alarm is also sent.

**Why edge-side and not server-side.** The autoencoder operates on raw features that legally cannot leave the client, so anomalous *data* (poisoned / drifted) is caught before it enters the federated channel — where only gradients are observable.

### 4.2 Stage B — Dynamic gradient validation (server)
The server computes, on the received deltas:

1. Robust norm baseline: $\hat{n} = \mathrm{median}_i \|\boldsymbol{\delta}_i\|_2$.
2. Norm clipping: $\boldsymbol{\delta}_i \leftarrow \boldsymbol{\delta}_i \cdot \min\bigl(1, \hat{n}/\|\boldsymbol{\delta}_i\|_2\bigr)$. Bounded magnitude defeats raw scaling attacks.
3. Reference direction $\boldsymbol{\delta}^{\star}$ = coordinate-wise median over $\{\boldsymbol{\delta}_i\}$. Median is a high-breakdown estimator that does *not* require a trusted dataset (unlike FLTrust).

### 4.3 Stage C — Context-aware aggregation
Per-client cosine to the reference:
$$\phi_i \;=\; \frac{\langle \boldsymbol{\delta}_i, \boldsymbol{\delta}^{\star}\rangle}{\|\boldsymbol{\delta}_i\|\,\|\boldsymbol{\delta}^{\star}\|}.$$

Trust score updated with EMA, $\alpha=0.6$:
$$\mathrm{TS}_i^{(t)} \;=\; \alpha\,\mathrm{TS}_i^{(t-1)} + (1-\alpha)\cdot \mathrm{ReLU}(\phi_i)\cdot(1-a_i).$$

Aggregation weight:
$$w_i^{(t)} \;=\; \frac{\mathrm{TS}_i^{(t)} \cdot \mathbb{1}[a_i = 0 \text{ OR } \phi_i \ge \phi_{\min}]}{\sum_{j\in K_t} \mathrm{TS}_j^{(t)} \cdot \mathbb{1}[\cdot]}.$$

Global update:
$$\boldsymbol{\theta}^{(t+1)} \;=\; \boldsymbol{\theta}^{(t)} + \sum_{i\in K_t} w_i^{(t)}\,\boldsymbol{\delta}_i.$$

The ReLU clips clients moving *opposite* to the consensus direction — the canonical signature of sign-flip attackers — without hard-rejecting Non-IID legitimate clients (which preserve $\phi_i \ge 0$ in expectation under the same global objective).

### 4.4 Lightweight design budget
- Edge AE: ≤16-unit bottleneck → ≤6k parameters; <2 ms / batch on ARM Cortex-A53.
- Server work per round: $O(m\,d)$ for medians (vs $O(m^2 d)$ for Krum). With $m=10, d=50\text{k}$ this is <1 ms on commodity CPU.
- Communication: identical to FedAvg (one delta + one byte alarm flag per client per round).

### 4.5 Stated assumptions
1. Server is honest-but-curious (does not collude).
2. At round 0 each client has a clean local sample large enough to fit the AE (≥256 samples).
3. The fraction of malicious clients $f/|K_t| < 0.5$ in expectation (otherwise the median is itself adversarial).
4. Local AE thresholds $\tau_i$ are bootstrapped *before* attacks begin — any pre-deployment compromise is out of scope.
5. Communication channels are authenticated (TLS); FEDShield does not address packet-level injection.
