"""Thin wrapper for the shared benchmark implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("_benchmark_common", _ROOT / "benchmark_common.py")
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("Cannot load shared benchmark_common.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

run_gp3_routing = _MODULE.run_gp3_routing
