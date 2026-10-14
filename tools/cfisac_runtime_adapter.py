#!/usr/bin/env python3
"""
CF-ISAC project runtime adapter.

This adapter contains NO surrogate mobility/channel/optimization physics.
It only constructs and forwards objects to the user's existing project.

Environment overrides (exact existing-project factories):
  CFISAC_SIM_FACTORY=module:function
  CFISAC_AP_FACTORY=module:function
  CFISAC_PREDICTOR_FACTORY=module:function
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _resolve(spec: str) -> Callable[..., Any]:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise RuntimeError(f"Invalid factory {spec!r}; expected module:function")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if not callable(fn):
        raise RuntimeError(f"{spec!r} does not resolve to a callable")
    return fn


def _load_config() -> dict[str, Any]:
    if yaml is None or not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, name))
    else:
        out[prefix] = obj
    return out


def _config_value(config: dict[str, Any], names: list[str], default: Any = None) -> Any:
    flat = _flatten(config)
    for name in names:
        if name in flat:
            return flat[name]
        if name in config:
            return config[name]
    return default


def _construct_with_aliases(cls: type, requested: dict[str, Any], label: str) -> Any:
    sig = inspect.signature(cls)
    aliases = {
        "num_users": ["num_users", "n_users", "K", "K_users"],
        "num_targets": ["num_targets", "n_targets", "Q", "Q_targets"],
        "area_size": ["area_size", "area", "side_length"],
        "dt": ["dt", "time_step", "delta_t"],
        "seed": ["seed", "random_seed"],
        "user_positions": ["user_positions", "users_positions", "positions_users"],
        "target_positions": ["target_positions", "targets_positions", "positions_targets"],
        "user_max_speed": ["user_max_speed", "user_speed", "users_speed", "speed_users"],
        "target_max_speed": ["target_max_speed", "target_speed", "targets_speed", "speed_targets"],
        "history_capacity": ["history_capacity", "history_size", "max_history"],
        "mobility_config": ["mobility_config", "mobility"],
        "mobility_model": ["mobility_model", "model"],
        "user_speed": ["user_speed", "users_speed", "speed_users"],
        "target_speed": ["target_speed", "targets_speed", "speed_targets"],
    }
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        found = False
        for source in aliases.get(name, [name]):
            if source in requested:
                kwargs[name] = requested[source]
                found = True
                break
        if not found and param.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"Cannot construct {label}: required parameter {name!r} is not covered. "
                f"Local signature: {sig}"
            )
    return cls(**kwargs)


def build_simulator(*, num_users: int, num_targets: int, area_size: float, dt: float, seed: int) -> Any:
    """Construct the project's existing mobility simulator; never synthesize one."""
    spec = os.getenv("CFISAC_SIM_FACTORY")
    if spec:
        return _resolve(spec)(
            num_users=num_users,
            num_targets=num_targets,
            area_size=area_size,
            dt=dt,
            seed=seed,
        )

    from src.simulation.mobility_simulator import MobilitySimulator

    config = _load_config()

    # The local project constructor requires initial user/target positions.
    # These are deterministic initial conditions only. All actual mobility
    # evolution, history, prediction inputs, and boundary handling remain in
    # the project's MobilitySimulator.
    rng = np.random.default_rng(seed)
    margin = 0.05 * area_size
    user_positions = rng.uniform(
        margin, area_size - margin, size=(num_users, 2)
    )
    target_positions = rng.uniform(
        margin, area_size - margin, size=(num_targets, 2)
    )

    requested = {
        "user_positions": user_positions,
        "target_positions": target_positions,
        "area_size": area_size,
        "dt": dt,
        "seed": seed,
        "user_max_speed": _config_value(
            config,
            [
                "mobility.user_max_speed",
                "mobility.users.max_speed",
                "user_max_speed",
            ],
        ),
        "target_max_speed": _config_value(
            config,
            [
                "mobility.target_max_speed",
                "mobility.targets.max_speed",
                "target_max_speed",
            ],
        ),
        "history_capacity": _config_value(
            config,
            [
                "mobility.history_capacity",
                "history_capacity",
            ],
        ),
        "mobility_config": _config_value(
            config,
            [
                "mobility_config",
                "simulation.mobility_config",
            ],
        ),
    }

    return _construct_with_aliases(
        MobilitySimulator,
        {k: v for k, v in requested.items() if v is not None},
        "MobilitySimulator",
    )

def _as_ap_positions(value: Any, num_aps: int) -> np.ndarray:
    candidate = getattr(value, "ap_positions", value)
    if isinstance(candidate, tuple) and candidate:
        candidate = candidate[0]
    arr = np.asarray(candidate, dtype=float)
    if arr.shape != (num_aps, 2):
        raise RuntimeError(f"Topology returned {arr.shape}; expected {(num_aps, 2)}")
    return arr


def build_ap_positions(*, num_aps: int, area_size: float, seed: int) -> np.ndarray:
    """Resolve AP topology from src.environment.topology only."""
    spec = os.getenv("CFISAC_AP_FACTORY")
    if spec:
        return _as_ap_positions(
            _resolve(spec)(num_aps=num_aps, area_size=area_size, seed=seed),
            num_aps,
        )

    import src.environment.topology as topology

    candidates = (
        "generate_ap_positions",
        "build_ap_positions",
        "create_ap_positions",
        "make_ap_positions",
        "generate_topology",
        "build_topology",
        "create_topology",
        "make_topology",
    )
    for name in candidates:
        fn = getattr(topology, name, None)
        if not callable(fn):
            continue
        sig = inspect.signature(fn)
        kwargs: dict[str, Any] = {}
        try:
            for pname, param in sig.parameters.items():
                if pname == "num_aps" or pname in ("M", "n_aps"):
                    kwargs[pname] = num_aps
                elif pname in ("area_size", "area", "side_length"):
                    kwargs[pname] = area_size
                elif pname in ("seed", "random_seed"):
                    kwargs[pname] = seed
                elif param.default is inspect.Parameter.empty:
                    raise TypeError
            return _as_ap_positions(fn(**kwargs), num_aps)
        except TypeError:
            continue

    for name in ("AP_POSITIONS", "ap_positions"):
        value = getattr(topology, name, None)
        if value is not None:
            return _as_ap_positions(value, num_aps)

    raise RuntimeError(
        "Could not resolve AP placement from src.environment.topology. "
        "Set CFISAC_AP_FACTORY to the project's existing topology factory."
    )


def build_predictor(kind: str = "cv") -> Any:
    """Use the project's public predictor factory."""
    spec = os.getenv("CFISAC_PREDICTOR_FACTORY")
    if spec:
        return _resolve(spec)(kind=kind)
    from src.prediction.predictor_interface import make_predictor
    return make_predictor(kind)
