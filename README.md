# EU-RAC

This repository contains the implementation used for the EU-RAC experiments in the paper. It includes the proposed EU-RAC method, representative stochastic routing baselines, network environments, OD selection utilities, and scripts for reproducing the reported experiments.

## Repository Layout

```text
eu-rac/
  README.md
  requirements.txt
  main.py
  eu_rac.py
  eu_rac_tabular.py
  benchmark.py
  benchmark_common.py
  evaluation.py
  func.py
  env.py
  od_selection.py
  Networks/
  data/
  sioux/
  Anaheim/
  Barcelona/
  Chicago/
  beijing/
  chengdu/
```

Temporary diagnostics, checkpoints, plotting previews, intermediate screening files, and historical development variants are excluded from the public release.

## Algorithms

- `eu_rac.py` implements the proposed neural EU-RAC algorithm used for large benchmark networks.
- `eu_rac_tabular.py` provides the tabular EU-RAC implementation used for Sioux Falls.
- `benchmark.py` contains the compact public dispatcher for EU-RAC smoke experiments.
- `evaluation.py` contains Monte Carlo evaluation and policy/path validation utilities.

The network-specific runners call into shared benchmark implementations for DOT, Robust, and GP3 where possible. The remaining network-specific runners cover the algorithms that are genuinely different across networks or rely on network-specific baselines. The release also includes the ablation entry points and the final uncertainty-environment files needed for the paper comparisons.

## Datasets and OD Pairs

This release includes the network files, uncertainty descriptions, and final OD files needed by the public entry points.

```text
Networks/Networks/SiouxFalls/
  SiouxFalls_network.csv
  SiouxFalls_0.4_random_sigma.npy
Networks/Networks/Anaheim/
  Anaheim_network.csv
  Anaheim_0.4_random_sigma.npy
Networks/Networks/Chicago_Sketch/
  Chicago_Sketch_network.csv
  Chicago_Sketch_0.4_random_sigma.npy
Barcelona/network/
  Barcelona_network.csv
  Barcelona_cov.npy
beijing/Beijing_network/
  Beijing_*.csv
  Beijing_Pairs.npy
  uncertainty/*.json
chengdu/Chengdu_network/
  *_network.csv
  uncertainty/*.json
data/
  */od_pairs.csv
```

For the paper experiments, the large benchmark networks use 50 representative OD pairs in total. The Beijing and Chengdu releases store the period labels in the public OD table, so the period-specific runners filter from the same published set. Sioux Falls uses a smaller representative OD set. OD pairs are screened by connectivity, feasible travel-time budget, and non-trivial routing uncertainty.

## Dependencies

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Some baselines require optional solvers:

- `pulp` for ILP-style baselines.
- `cvxopt` for legacy helper functions.
- A working PyTorch installation for neural EU-RAC and neural baselines.

## Quick Start

Run a smoke test:

```bash
python main.py --network sioux --algorithm eurac-tabular --od-file data/sioux/od_pairs.csv --episodes 1 --eval-episodes 1
```

For a large network, use the neural EU-RAC implementation:

```bash
python main.py --network chicago --algorithm eurac --od-file data/chicago/od_pairs.csv --episodes 1 --eval-episodes 1
```

## Reproducing Experiments

The main experiments use the selected OD pairs and budget settings reported in the paper. A typical command follows this pattern:

```bash
python main.py --network anaheim --algorithm eurac --od-file data/anaheim/od_pairs.csv
```

Use `od_selection.py` to generate new screened OD lists when needed. The released `data/*/od_pairs.csv` files are the authoritative OD sets for reproducing the paper tables.

## Code Style

All public comments, docstrings, command-line help text, and documentation are written in English. Before publishing, run a non-ASCII scan over the final source files and remove any corrupted comments or encoding artifacts.

## Notes

Large raw outputs, checkpoints, temporary diagnostics, and plotting previews are intentionally excluded from the repository. They can be regenerated from the released code and selected data.
