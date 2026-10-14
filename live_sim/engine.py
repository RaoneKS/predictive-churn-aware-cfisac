from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.clustering.churn import max_raw_churn_cost
from src.environment.mobility_configs import STOCHASTIC_CONFIG
from src.environment.topology import generate_ap_positions, generate_positions
from src.optimization.baselines import cluster_for_positions
from src.optimization.risk_aware import risk_aware_slow_update
from src.optimization.two_timescale import fast_update
from src.prediction.predictor_interface import make_predictor
from src.simulation.mobility_simulator import MobilitySimulator


@dataclass
class LiveConfig:
    num_aps: int = 20
    num_users: int = 5
    num_targets: int = 2
    num_antennas: int = 4
    area_size: float = 500.0
    dt: float = 1.0
    user_max_speed: float = 15.0
    target_max_speed: float = 10.0
    mobility_mode: str = "constant_velocity"
    T_slow: int = 5
    prediction_horizon: int = 5
    predictor_type: str = "cv"
    history_length: int = 10
    P_max: float = 1.0
    precoder: str = "mrt"
    alpha_power: float = 1.0
    beta_power: float = 1.0
    num_scenarios: int = 100
    cvar_alpha: float = 0.9
    lambda_churn: float = 1.0
    seed: int = 42
    base_fast_seed: int = 7
    aps_per_user: int = 3
    tx_aps_per_target: int = 2
    rx_aps_per_target: int = 2
    p1_mode: bool = False
    p1_max_candidates: int = 20
    enforce_deficiency_cvar: bool = False
    max_comm_cvar: float = 1e6
    max_sensing_cvar: float = 1e-9
    min_rate_bps: float = 1e6
    epsilon_trk: float = 1e-9


@dataclass
class LiveCFISACEngine:
    """One mutable simulation state; no precomputed trajectory is stored."""

    cfg: LiveConfig
    sim: MobilitySimulator
    ap_positions: np.ndarray
    predictor: Any
    x: np.ndarray
    y_tx: np.ndarray
    y_rx: np.ndarray
    step: int = 0
    slow_epoch: int = 0
    reconfigurations: int = 0
    churn_total: float = 0.0
    records: list[dict[str, Any]] = field(default_factory=list)
    last_slow: dict[str, Any] | None = None
    last_fast: dict[str, Any] | None = None

    @classmethod
    def create(cls, cfg: LiveConfig) -> "LiveCFISACEngine":
        if cfg.T_slow < 1 or cfg.prediction_horizon < 1:
            raise ValueError("T_slow and prediction_horizon must be >= 1")
        if cfg.num_scenarios < 0:
            raise ValueError("num_scenarios must be >= 0")

        ap = generate_ap_positions(cfg.num_aps, cfg.area_size, seed=cfg.seed)
        users = generate_positions(cfg.num_users, cfg.area_size, seed=cfg.seed + 1)
        targets = generate_positions(cfg.num_targets, cfg.area_size, seed=cfg.seed + 2)

        mobility_config = None
        if cfg.mobility_mode == "stochastic":
            mobility_config = STOCHASTIC_CONFIG
        elif cfg.mobility_mode != "constant_velocity":
            raise ValueError(f"Unknown mobility_mode: {cfg.mobility_mode}")

        sim = MobilitySimulator(
            users,
            targets,
            area_size=cfg.area_size,
            dt=cfg.dt,
            user_max_speed=cfg.user_max_speed,
            target_max_speed=cfg.target_max_speed,
            seed=cfg.seed + 3,
            mobility_config=mobility_config,
            history_capacity=max(50, cfg.history_length + cfg.prediction_horizon + 5),
        )

        predictor = cls._make_predictor(cfg)
        x, y_tx, y_rx = cluster_for_positions(
            ap,
            sim.user_positions,
            sim.target_positions,
            aps_per_user=cfg.aps_per_user,
            tx_aps_per_target=cfg.tx_aps_per_target,
            rx_aps_per_target=cfg.rx_aps_per_target,
        )

        return cls(cfg, sim, ap, predictor, x, y_tx, y_rx)

    @staticmethod
    def _make_predictor(cfg: LiveConfig):
        if cfg.predictor_type == "cv":
            return make_predictor("cv")
        root = Path(__file__).resolve().parents[1]
        user_ckpt = root / "results" / "prediction" / "lstm_users.pt"
        target_ckpt = root / "results" / "prediction" / "lstm_targets.pt"
        if not user_ckpt.exists() or not target_ckpt.exists():
            raise FileNotFoundError(
                "LSTM predictor selected, but results/prediction/lstm_users.pt "
                "or lstm_targets.pt is missing."
            )
        return make_predictor(
            "lstm",
            user_ckpt_path=str(user_ckpt),
            target_ckpt_path=str(target_ckpt),
            area_size=cfg.area_size,
            history_length=cfg.history_length,
        )

    @property
    def current_positions(self) -> dict[str, np.ndarray]:
        return {
            "users": self.sim.user_positions.copy(),
            "targets": self.sim.target_positions.copy(),
            "aps": self.ap_positions.copy(),
        }

    def _ref(self) -> dict[str, float]:
        return {
            "churn_events": max_raw_churn_cost(
                self.cfg.num_users,
                self.cfg.num_targets,
                aps_per_user=self.cfg.aps_per_user,
                tx_aps_per_target=self.cfg.tx_aps_per_target,
                rx_aps_per_target=self.cfg.rx_aps_per_target,
                num_aps=self.cfg.num_aps,
            )
        }

    def _slow_update(self) -> dict[str, Any]:
        if getattr(self.cfg, "p1_mode", False):
            from src.optimization.p1_solver import solve_p1_decomposed
            eps_trk = __import__("numpy").full(self.cfg.num_targets, getattr(self.cfg, "epsilon_trk", 1e-9)) if getattr(self.cfg, "epsilon_trk", None) is not None else None
            upd = solve_p1_decomposed(
                sim=self.sim,
                predictor=self.predictor,
                ap_positions=self.ap_positions,
                num_antennas=self.cfg.num_antennas,
                prev_x=self.x,
                prev_y_tx=self.y_tx,
                prev_y_rx=self.y_rx,
                T_slow=self.cfg.T_slow,
                P_max=self.cfg.P_max,
                precoder=self.cfg.precoder,
                alpha_power=self.cfg.alpha_power,
                beta_power=self.cfg.beta_power,
                cluster_kwargs={
                    "aps_per_user": self.cfg.aps_per_user,
                    "tx_aps_per_target": self.cfg.tx_aps_per_target,
                    "rx_aps_per_target": self.cfg.rx_aps_per_target,
                },
                perf_kwargs={},
                churn_kwargs={"lambda_churn": self.cfg.lambda_churn},
                phys_kwargs={},
                sens_kwargs={},
                joint_kwargs={
                    "enforce_qos": getattr(self.cfg, "enforce_qos", False),
                    "min_rate": getattr(self.cfg, "min_rate_bps", None),
                },
                ref=self._ref(),
                p1_max_candidates=getattr(self.cfg, "p1_max_candidates", 20),
                seed=self.cfg.seed + self.slow_epoch,
                horizon_aggregation="sum",
                enforce_deficiency_cvar=getattr(self.cfg, "enforce_deficiency_cvar", False),
                max_comm_cvar=getattr(self.cfg, "max_comm_cvar", None),
                max_sensing_cvar=getattr(self.cfg, "max_sensing_cvar", None),
                min_rate_bps=getattr(self.cfg, "min_rate_bps", None),
                epsilon_trk=eps_trk,
                enforce_qos=getattr(self.cfg, "enforce_qos", False),
                num_scenarios=self.cfg.num_scenarios,
                scenario_seed=self.cfg.seed + self.slow_epoch
            )
            # Adapt upd structure to look like decision_info for _record
            upd["decision_info"] = {
                "decision": "P1_RECONFIGURE" if not upd["fallback_used"] else "FALLBACK",
                "raw_churn": upd["churn"],
                "net_gain": upd["objective"],
                "predicted_gain": upd["objective"],
                "cvar_rejected": False,
                "deficiency_cvar_results": upd.get("deficiency_cvar_results")
            }
        else:
            upd = risk_aware_slow_update(
                self.sim,
                self.predictor,
                self.ap_positions,
                self.x,
                self.y_tx,
                self.y_rx,
                self.cfg.T_slow,
                num_scenarios=self.cfg.num_scenarios,
                alpha=self.cfg.cvar_alpha,
                scenario_seed=self.cfg.seed + self.slow_epoch,
                cluster_kwargs={
                    "aps_per_user": self.cfg.aps_per_user,
                    "tx_aps_per_target": self.cfg.tx_aps_per_target,
                    "rx_aps_per_target": self.cfg.rx_aps_per_target,
                },
                churn_kwargs={"lambda_churn": self.cfg.lambda_churn},
                ref=self._ref(),
                horizon_aggregation="sum",
                enforce_deficiency_cvar=getattr(self.cfg, "enforce_deficiency_cvar", False),
                max_comm_cvar=getattr(self.cfg, "max_comm_cvar", None),
                max_sensing_cvar=getattr(self.cfg, "max_sensing_cvar", None),
                min_rate_bps=getattr(self.cfg, "min_rate_bps", None),
                epsilon_trk=__import__("numpy").full(self.cfg.num_targets, getattr(self.cfg, "epsilon_trk", 1e-9)) if getattr(self.cfg, "enforce_deficiency_cvar", False) else None,
            )

        self.x = np.asarray(upd["x"], dtype=int)
        self.y_tx = np.asarray(upd["y_tx"], dtype=int)
        self.y_rx = np.asarray(upd["y_rx"], dtype=int)
        self.last_slow = upd
        self.slow_epoch += 1

        info = upd["decision_info"]
        if info["decision"] in ["RECONFIGURE", "P1_RECONFIGURE"]:
            self.reconfigurations += 1
            self.churn_total += float(info["raw_churn"])
        return upd

    def _fast_update(self) -> dict[str, Any]:
        res = fast_update(
            self.ap_positions,
            self.sim.user_positions,
            self.sim.target_positions,
            self.x,
            self.y_tx,
            self.y_rx,
            self.cfg.num_antennas,
            self.cfg.P_max,
            precoder=self.cfg.precoder,
            alpha=self.cfg.alpha_power,
            beta=self.cfg.beta_power,
            fast_seed=self.cfg.base_fast_seed * 1_000_003 + self.step,
        )
        self.last_fast = res
        return res

    def evaluate_current(self) -> dict[str, Any]:
        """Evaluate the current physical state without advancing it."""
        fast = self._fast_update()
        return self._record(fast, self.last_slow, advanced=False)

    def step_once(self) -> dict[str, Any]:
        """Advance exactly one mobility block, then run the real pipeline."""
        self.sim.step()
        self.step += 1
        slow = None
        if self.step % self.cfg.T_slow == 0:
            slow = self._slow_update()
        fast = self._fast_update()
        return self._record(fast, slow, advanced=True)

    def _record(self, fast: dict[str, Any], slow: dict[str, Any] | None, advanced: bool) -> dict[str, Any]:
        decision_info = (slow or self.last_slow or {}).get("decision_info", {})
        pred_users = (slow or self.last_slow or {}).get("pred_users")
        pred_targets = (slow or self.last_slow or {}).get("pred_targets")
        p1_mode = getattr(self.cfg, "p1_mode", False)
        p1_obj = (slow or self.last_slow or {}).get("objective", 0.0) if p1_mode else 0.0
        p1_candidates = (slow or self.last_slow or {}).get("num_candidates_generated", 0) if p1_mode else 0
        p1_feasible = p1_candidates - sum((slow or self.last_slow or {}).get("rejection_counts", {}).values()) if p1_mode else 0
        record = {
            "step": self.step,
            "advanced": advanced,
            "decision": decision_info.get("decision", "BOOTSTRAP"),
            "raw_churn": float(decision_info.get("raw_churn", 0.0)),
            "net_gain": float(decision_info.get("net_gain", 0.0)),
            "predicted_gain": float(decision_info.get("predicted_gain", 0.0)),
            "deterministic_gain": float((slow or self.last_slow or {}).get("deterministic_gain", 0.0)),
            "risk_adjusted_gain": float((slow or self.last_slow or {}).get("risk_adjusted_gain", 0.0)),
            "var": float((slow or self.last_slow or {}).get("var") or 0.0),
            "cvar": float((slow or self.last_slow or {}).get("cvar") or 0.0),
            "cvar_rejected": bool(decision_info.get("cvar_rejected", False)),
            "deficiency_cvar_results": decision_info.get("deficiency_cvar_results", None),
            "rate_bps": float(np.sum(fast["rates_bps"])),
            "min_user_rate_bps": float(fast["min_user_rate_bps"]),
            "sensing_info": float(np.sum(fast["information_gain"])),
            "objective": float(fast["objective"]),
            "P_comm": np.asarray(fast["P_comm"], dtype=float).copy(),
            "P_sens": np.asarray(fast["P_sens"], dtype=float).copy(),
            "P_tx": np.asarray(fast["P_tx"], dtype=float).copy(),
            "feasible": bool(fast["feasible"]),
            "max_power_violation": float(fast["max_power_violation"]),
            "users": self.sim.user_positions.copy(),
            "targets": self.sim.target_positions.copy(),
            "aps": self.ap_positions.copy(),
            "x": self.x.copy(),
            "y_tx": self.y_tx.copy(),
            "y_rx": self.y_rx.copy(),
            "pred_users": None if pred_users is None else np.asarray(pred_users).copy(),
            "pred_targets": None if pred_targets is None else np.asarray(pred_targets).copy(),
            "reconfigurations": self.reconfigurations,
            "churn_total": self.churn_total,
            "p1_mode": p1_mode,
            "p1_candidates": p1_candidates,
            "p1_feasible": p1_feasible,
            "p1_objective": p1_obj,
            "p1_comm_util": (slow or self.last_slow or {}).get("comm_utility", 0.0) if p1_mode else 0.0,
            "p1_sens_util": (slow or self.last_slow or {}).get("sensing_utility", 0.0) if p1_mode else 0.0,
            "p1_energy": (slow or self.last_slow or {}).get("energy", 0.0) if p1_mode else 0.0,
            "p1_fronthaul": (slow or self.last_slow or {}).get("fronthaul", 0.0) if p1_mode else 0.0,
            "p1_fallback": (slow or self.last_slow or {}).get("fallback_used", False) if p1_mode else False,
        }
        self.records.append(record)
        return record

    def nudge_user(self, index: int, dx: float, dy: float) -> None:
        self._nudge(self.sim.user_positions, index, dx, dy)

    def nudge_target(self, index: int, dx: float, dy: float) -> None:
        self._nudge(self.sim.target_positions, index, dx, dy)

    def _nudge(self, arr: np.ndarray, index: int, dx: float, dy: float) -> None:
        if not 0 <= index < len(arr):
            raise IndexError(index)
        arr[index, 0] = np.clip(arr[index, 0] + dx, 0.0, self.cfg.area_size)
        arr[index, 1] = np.clip(arr[index, 1] + dy, 0.0, self.cfg.area_size)

    def history_table(self):
        import pandas as pd
        rows = []
        for r in self.records:
            rows.append({k: r[k] for k in (
                "step", "decision", "raw_churn", "net_gain", "predicted_gain",
                "risk_adjusted_gain", "var", "cvar", "rate_bps",
                "min_user_rate_bps", "sensing_info", "objective", "feasible",
                "max_power_violation", "reconfigurations", "churn_total",
            )})
        return pd.DataFrame(rows)
