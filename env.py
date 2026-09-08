"""Environment loading utilities for the public EU-RAC entry points."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

_ENV_SPECS = {
    "sioux": ("sioux/sioux_env.py", "SiouxEnv"),
    "siouxfalls": ("sioux/sioux_env.py", "SiouxEnv"),
    "anaheim": ("Anaheim/anaheim_env.py", "AnaheimEnv"),
    "barcelona": ("Barcelona/barcelona_env.py", "BarcelonaEnv"),
    "chicago": ("Chicago/chicago_env.py", "ChicagoEnv"),
    "beijing": ("beijing/beijing_env.py", "BeijingEnv"),
    "chengdu": ("chengdu/chengdu_env.py", "ChengduEnv"),
}


def _load_class(module_path: Path, class_name: str):
    """Load a class from a source file without requiring package installation."""
    module_dir = str(module_path.parent)
    module_name = f"_eurac_public_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_sys_path = list(sys.path)
    try:
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_sys_path
    return getattr(module, class_name)


def normalize_network(name: str) -> str:
    """Return a canonical network key."""
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "siouxfalls": "sioux",
        "sioux": "sioux",
        "anaheim": "anaheim",
        "barcelona": "barcelona",
        "chicago": "chicago",
        "chicagosketch": "chicago",
        "beijing": "beijing",
        "chengdu": "chengdu",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported network {name!r}. Supported networks: {sorted(set(aliases.values()))}")
    return aliases[key]


def make_env(network: str, origin: int, destination: int, budget: float, **kwargs: Any):
    """Construct a routing environment for a benchmark network."""
    key = normalize_network(network)
    rel_path, class_name = _ENV_SPECS[key]
    env_class = _load_class(ROOT / rel_path, class_name)
    if key in {"beijing", "chengdu"}:
        return env_class(origin=origin, destination=destination, budget=budget, **kwargs)
    return env_class(origin=origin, dest=destination, budget=budget, **kwargs)
