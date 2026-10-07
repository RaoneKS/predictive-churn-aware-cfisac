#!/usr/bin/env python3
"""Diagnose the existing CF-ISAC runtime API; read-only."""
from __future__ import annotations
import inspect
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULES = [
    "src.simulation.mobility_simulator",
    "src.environment.topology",
    "src.environment.scenario",
    "src.prediction.predictor_interface",
    "src.optimization.risk_aware",
    "src.optimization.two_timescale",
]

for name in MODULES:
    print(f"\n=== {name} ===")
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        print(f"IMPORT_ERROR: {exc}")
        continue
    for attr_name in sorted(dir(mod)):
        if attr_name.startswith("_"):
            continue
        obj = getattr(mod, attr_name)
        if inspect.isclass(obj) and getattr(obj, "__module__", "") == name:
            try: sig = inspect.signature(obj)
            except Exception: sig = "(signature unavailable)"
            print(f"CLASS  {attr_name}{sig}")
        elif inspect.isfunction(obj) and getattr(obj, "__module__", "") == name:
            try: sig = inspect.signature(obj)
            except Exception: sig = "(signature unavailable)"
            print(f"FUNC   {attr_name}{sig}")
