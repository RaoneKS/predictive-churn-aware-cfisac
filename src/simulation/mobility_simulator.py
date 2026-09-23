import numpy as np

from src.environment.mobility import update_positions, generate_velocities
from src.environment.stochastic_mobility import StochasticMobilityGroup


class MobilitySimulator:
    """
    Simulates mobile communication users and sensing targets.

    Mobility mode is selected via the `mobility_config` dict:

    constant_velocity (default, backward-compatible)
        Velocity is fixed after initialisation.
        Reflective boundaries.
        All callers that do not pass mobility_config get this mode unchanged.

    stochastic
        Correlated-velocity (OU-process) model:
            v_{t+1} = rho_v * v_t + (1-rho_v)*v_pref + sigma_v*eps_t
        Speed clamped to max_speed.  Reflective boundaries.

    Parameters
    ----------
    user_positions    : (N_u, 2)
    target_positions  : (N_t, 2)
    area_size         : float
    dt                : float
    user_max_speed    : float  (constant_velocity mode only)
    target_max_speed  : float  (constant_velocity mode only)
    seed              : int
    mobility_config   : dict or None — None → constant_velocity
    history_capacity  : int — max recent (user_pos, target_pos) steps to keep
    """

    def __init__(
        self,
        user_positions,
        target_positions,
        area_size=500.0,
        dt=1.0,
        user_max_speed=15.0,
        target_max_speed=10.0,
        seed=42,
        mobility_config=None,
        history_capacity=50,
    ):
        self.user_positions   = np.asarray(user_positions,   dtype=float).copy()
        self.target_positions = np.asarray(target_positions, dtype=float).copy()
        self.area_size        = area_size
        self.dt               = dt
        self.history_capacity = history_capacity
        self.position_history = []   # list of (user_pos_copy, target_pos_copy)

        # Genuine boundary-reflection event counters (not velocity-sign-flip
        # based). Incremented only when a position actually contacted and
        # reflected off the simulation-area boundary this step.
        self.user_boundary_event_counts   = np.zeros(len(self.user_positions),   dtype=int)
        self.target_boundary_event_counts = np.zeros(len(self.target_positions), dtype=int)

        mob_type = "constant_velocity"
        if mobility_config is not None:
            mob_type = mobility_config.get("type", "constant_velocity")
        self._mob_type = mob_type

        if mob_type == "stochastic" and mobility_config is not None:
            u_cfg = mobility_config.get("users", {})
            t_cfg = mobility_config.get("targets", {})
            # Independent RNGs for users and targets
            u_rng = np.random.default_rng(seed)
            t_rng = np.random.default_rng(seed + 1)

            self._user_group = StochasticMobilityGroup(
                positions=self.user_positions,
                rng=u_rng,
                max_speed=float(u_cfg.get("max_speed",       15.0)),
                preferred_speed=float(u_cfg.get("preferred_speed", 8.0)),
                rho_v=float(u_cfg.get("rho_v",               0.85)),
                sigma_v=float(u_cfg.get("sigma_v",           2.0)),
                area_size=area_size,
                dt=dt,
                pref_refresh_interval=int(
                    u_cfg.get("pref_refresh_interval", 20)
                ),
            )
            self._target_group = StochasticMobilityGroup(
                positions=self.target_positions,
                rng=t_rng,
                max_speed=float(t_cfg.get("max_speed",       10.0)),
                preferred_speed=float(t_cfg.get("preferred_speed", 5.0)),
                rho_v=float(t_cfg.get("rho_v",               0.75)),
                sigma_v=float(t_cfg.get("sigma_v",           3.0)),
                area_size=area_size,
                dt=dt,
                pref_refresh_interval=int(
                    t_cfg.get("pref_refresh_interval", 20)
                ),
            )
            self.user_velocities   = self._user_group.velocities.copy()
            self.target_velocities = self._target_group.velocities.copy()

        else:
            # ── Backward-compatible constant-velocity initialisation ──────────
            self._user_group   = None
            self._target_group = None
            self.user_velocities = generate_velocities(
                len(self.user_positions), user_max_speed, seed=seed
            )
            self.target_velocities = generate_velocities(
                len(self.target_positions), target_max_speed, seed=seed + 1
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Core step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self):
        """Advance users and targets by one time block."""
        if self._mob_type == "stochastic":
            self.user_positions,   self.user_velocities   = self._user_group.step()
            self.target_positions, self.target_velocities = self._target_group.step()
            user_hit   = self._user_group.last_boundary_hit()
            target_hit = self._target_group.last_boundary_hit()
        else:
            self.user_positions, self.user_velocities, user_hit = update_positions(
                self.user_positions, self.user_velocities, self.dt, self.area_size,
                return_events=True,
            )
            self.target_positions, self.target_velocities, target_hit = update_positions(
                self.target_positions, self.target_velocities, self.dt, self.area_size,
                return_events=True,
            )

        if user_hit is not None:
            self.user_boundary_event_counts += user_hit.astype(int)
        if target_hit is not None:
            self.target_boundary_event_counts += target_hit.astype(int)

        # Maintain rolling position history for LSTM input
        self.position_history.append(
            (self.user_positions.copy(), self.target_positions.copy())
        )
        if len(self.position_history) > self.history_capacity:
            self.position_history.pop(0)

        return self.user_positions.copy(), self.target_positions.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Constant-velocity predictions (used by existing PREDICTIVE policy — unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_users(self, horizon):
        """Constant-velocity prediction for users. Returns (N_u, H, 2)."""
        predictions = []
        for position, velocity in zip(self.user_positions, self.user_velocities):
            future = np.asarray([
                position + h * velocity * self.dt
                for h in range(1, horizon + 1)
            ])
            predictions.append(future)
        return np.asarray(predictions)

    def predict_targets(self, horizon):
        """Constant-velocity prediction for targets. Returns (N_t, H, 2)."""
        predictions = []
        for position, velocity in zip(self.target_positions, self.target_velocities):
            future = np.asarray([
                position + h * velocity * self.dt
                for h in range(1, horizon + 1)
            ])
            predictions.append(future)
        return np.asarray(predictions)

    # ─────────────────────────────────────────────────────────────────────────
    # History accessors (for LSTM input construction)
    # ─────────────────────────────────────────────────────────────────────────

    def get_user_history(self, history_length: int):
        """
        Return recent user positions as (num_users, history_length, 2).
        Returns None if fewer than history_length steps have been recorded.
        """
        if len(self.position_history) < history_length:
            return None
        hist = self.position_history[-history_length:]
        return np.stack([h[0] for h in hist], axis=1)   # (N_u, L, 2)

    def get_target_history(self, history_length: int):
        """
        Return recent target positions as (num_targets, history_length, 2).
        Returns None if fewer than history_length steps have been recorded.
        """
        if len(self.position_history) < history_length:
            return None
        hist = self.position_history[-history_length:]
        return np.stack([h[1] for h in hist], axis=1)   # (N_t, L, 2)
