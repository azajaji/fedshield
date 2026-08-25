# FedShield

Companion code for the paper:

> **FedShield: Configurable Mean–Median Robust Aggregation for Poisoning-Resilient Federated Healthcare-IoT Learning**
> Abdulaziz Saleh Alajaji, Maryam Jamal Alablani, and Mohammad Mehedi Hassan
> *IEEE Access*, 2026.

FedShield is a configurable Byzantine-robust aggregation rule for federated learning. It ranks
client updates with a Krum-style score, then combines a narrow top-2 mean with a broader top-7
coordinate-wise median through a single convex mixing weight `alpha` that is fixed per deployment.

This repository contains the aggregators, the attack catalogue, the per-dataset configurations,
the fixed seed list, the per-seed result files, and the scripts that regenerate every table and
figure in the paper.

## Repository layout

```
fedshield/
  config.py         # ExperimentConfig dataclasses; YAML/JSON I/O
  data_loader.py    # dataset loaders + Non-IID partitioning
  models.py         # 1D-CNN (ECG), 12-channel 1D-CNN, IoMT MLP, edge autoencoder
  attacks.py        # data-layer, update-layer, and post-collection adaptive attacks
  defenses.py       # FedAvg, Krum, Multi-Krum, Trimmed-Mean, RFA, DnC, FLTrust,
                    # FoolsGold, Bulyan, coordinate-wise median, FedShield
  federated.py      # FederatedTrainer orchestrator
  evaluation.py     # cross-experiment aggregation
  visualize.py      # figure generation
  main.py           # CLI entry point
configs/            # per-dataset experiment configurations
scripts/            # dataset preparation, experiment runners, table/figure generation
docs/               # technical summary, experimental design, results analysis
```

This is a minimal archive: the method, the baselines, the attacks, the configurations, and the
scripts that produce the results. Run outputs are not committed, since running the pipeline
regenerates them. The published values are the tables in the article.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

The four benchmarks used in the paper are **not redistributed here**. Each must be obtained from
its original provider under that provider's terms:

| Dataset | Source |
|---|---|
| MIT-BIH Arrhythmia | <https://physionet.org/content/mitdb/1.0.0/> |
| PTB-XL | <https://physionet.org/content/ptb-xl/> |
| PhysioNet/CinC Challenge 2017 | <https://physionet.org/content/challenge-2017/> |
| CIC-IoMT-2024 | Canadian Institute for Cybersecurity |

Preprocessing scripts for each are under `scripts/` (`prep_mitbih.py`, `prep_ptbxl.py`,
`prep_physionet_2017.py`, `prep_ciciomt.py`). Place the preprocessed arrays under `data/<name>/`.

A structured synthetic fallback preserving the federation topology is available so the pipeline
runs end to end without external data; it is not the basis for any published result.

## Reproducing the paper

The headline matrix is 33 cells per dataset (one clean cell plus 32 attacked cells: eight attack
types by four malicious-client ratios), five seeds, and nine method configurations. The five
seeds are `{0, 7, 13, 21, 42}`.

```bash
python -m fedshield.main --config configs/mitbih_fedshield.yaml   # single configuration
python scripts/run_attack_matrix.py                               # full headline matrix
python scripts/run_ablations.py                                   # ablations
python scripts/compile_results.py                                 # aggregate per-seed runs
python scripts/make_results_tables.py                             # tables
python scripts/make_figures.py                                    # figures
```

Runs write per-seed metric files under `results/`, which `compile_results.py` aggregates into the
table and figure inputs.

## Configuration

FedShield's locked settings in the paper are `m_2 = 2` (mean support), `m_med = 7` (median
support), Byzantine tolerance `f = 2`, and the two mixing-weight anchors `alpha = 0.25` and
`alpha = 0.90`. The main results use the core aggregation rule only. The optional preprocessing
diagnostics (edge autoencoder alarm, norm-MAD clipping, reference-direction cosine) are disabled
and are evaluated separately in the ablation.

## Scope

Results are scoped to the evaluated non-adaptive attacks. The paper documents two cases where the
distance-based prefilter is evaded — centroid-near mimicry and an adaptive Krum-survival attack —
under which plain averaging is the stronger response. PTB-XL is a documented exception on which
broad averaging remains preferable.

## Citation

```bibtex
@article{alajaji2026fedshield,
  author  = {Alajaji, Abdulaziz Saleh and Alablani, Maryam Jamal and Hassan, Mohammad Mehedi},
  title   = {{FedShield}: Configurable Mean--Median Robust Aggregation for
             Poisoning-Resilient Federated Healthcare-IoT Learning},
  journal = {IEEE Access},
  year    = {2026}
}
```

## Funding

The authors extend their appreciation to the Deanship of Scientific Research at King Saud
University for funding this work through the Waed Program (W25-83).

## License

Released under the MIT License. See [LICENSE](LICENSE).
