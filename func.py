"""Shared utility functions for EU-RAC experiments."""

from __future__ import annotations

import csv
from pathlib import Path


def read_od_pairs(path: str | Path) -> list[tuple[int, int]]:
    """Read OD pairs from a CSV file with origin/destination columns or two columns."""
    od_path = Path(path)
    pairs: list[tuple[int, int]] = []
    with od_path.open(newline="", encoding="utf-8") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        if has_header:
            reader = csv.DictReader(handle)
            for row in reader:
                origin = row.get("origin") or row.get("Origin") or row.get("source") or row.get("Source")
                dest = row.get("destination") or row.get("Destination") or row.get("dest") or row.get("Dest")
                if origin is None or dest is None:
                    values = list(row.values())
                    origin, dest = values[0], values[1]
                pairs.append((int(origin), int(dest)))
        else:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) >= 2:
                    pairs.append((int(row[0]), int(row[1])))
    return pairs


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
