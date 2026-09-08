# OD Selection Files

This directory contains the final OD files intended for the public EU-RAC release. Network CSV/NPY files are stored in the paths used directly by the environment modules, such as `Networks/Networks/*`, `Barcelona/network`, `beijing/Beijing_network`, and `chengdu/Chengdu_network`.

## File Status

- `sioux/od_pairs.csv`: representative Sioux Falls OD set used by the small-network experiments.
- `anaheim/od_pairs.csv`: 50-pair Anaheim OD set.
- `barcelona/od_pairs.csv`: 50-pair Barcelona OD set.
- `chicago/od_pairs.csv`: 50-pair Chicago Sketch OD set.
- `beijing/od_pairs.csv`: 50-pair Beijing OD table with period labels.
- `chengdu/od_pairs.csv`: 50-pair Chengdu OD table with period labels.

## Public Benchmark Policy

The public release keeps only the final OD files used in the paper. Large static networks use 50 representative OD pairs in total. Beijing and Chengdu store the period labels in the released OD table so the period-specific runners can filter from the same public set. Sioux Falls uses a smaller representative OD set. Use `od_selection.py` only when generating a new screened OD set from the included network data.

All OD CSV files use at least these columns:

```text
origin,destination
```

Additional metadata columns may be included when they are useful for auditing OD selection.
