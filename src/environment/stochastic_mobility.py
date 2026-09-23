"""
Correlated-velocity (Ornstein-Uhlenbeck-inspired) mobility model.

v_{t+1} = rho_v * v_t
        + (1-rho_v) * v_pref(t)
        + sigma_v * eps_t

p_{t+1} = p_t + v_{t+1} * dt

The model provides temporally correlated, nonlinear trajectories while
remaining lightweight and reproducible.
"""

import numpy as np


def _reflect(
    positions: np.ndarray,
    velocities: np.ndarray,
    area_size: float,
):
    """
    Reflect positions and velocity components at square boundaries.

    Uses a loop so large overshoots are handled robustly even when a step
    crosses a boundary by more than one area width.

    Returns
    -------
    positions  : reflected positions, same shape as input
    velocities : reflected velocities, same shape as input
    boundary_hit : bool ndarray, shape (N,) — True for any object that had
        at least one reflection event (in x and/or y) during this call.
        This is a genuine geometric event indicator (a boundary was
        actually contacted), independent of velocity sign flips caused by
        stochastic noise elsewhere in the update.
    """
    positions = np.asarray(positions, dtype=float).copy()
    velocities = np.asarray(velocities, dtype=float).copy()
    boundary_hit = np.zeros(len(positions), dtype=bool)

    for i in range(len(positions)):
        for d in range(2):
            # Robust reflection for arbitrary overshoot.
            while positions[i, d] < 0.0 or positions[i, d] > area_size:
                boundary_hit[i] = True
                if positions[i, d] < 0.0:
                    positions[i, d] = -positions[i, d]
                    velocities[i, d] *= -1.0
                elif positions[i, d] > area_size:
                    positions[i, d] = 2.0 * area_size - positions[i, d]
                    velocities[i, d] *= -1.0

    return positions, velocities, boundary_hit


class StochasticMobilityGroup:
    """
    Group mobility model for users or sensing targets.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Initial positions in metres.
    rng : np.random.Generator
        Caller-controlled random generator.
    max_speed : float
        Hard speed ceiling in m/block.
    preferred_speed : float
        Typical preferred velocity magnitude in m/block.
    rho_v : float
        Velocity persistence, strictly between 0 and 1.
    sigma_v : float
        Standard deviation of additive velocity noise.
    area_size : float
        Square simulation-area side length in metres.
    dt : float
        Time-step duration.
    pref_refresh_interval : int
        Number of blocks between preferred-direction updates.
    """

    def __init__(
        self,
        positions: np.ndarray,
        rng: np.random.Generator,
        max_speed: float = 15.0,
        preferred_speed: float = 8.0,
        rho_v: float = 0.85,
        sigma_v: float = 2.0,
        area_size: float = 500.0,
        dt: float = 1.0,
        pref_refresh_interval: int = 20,
    ):
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")

        if max_speed <= 0:
            raise ValueError("max_speed must be > 0")

        if preferred_speed < 0:
            raise ValueError("preferred_speed must be >= 0")

        if not 0.0 < rho_v < 1.0:
            raise ValueError("rho_v must satisfy 0 < rho_v < 1")

        if sigma_v < 0:
            raise ValueError("sigma_v must be >= 0")

        if area_size <= 0:
            raise ValueError("area_size must be > 0")

        if dt <= 0:
            raise ValueError("dt must be > 0")

        if pref_refresh_interval <= 0:
            raise ValueError("pref_refresh_interval must be > 0")

        self.positions = np.asarray(positions, dtype=float).copy()

        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (N, 2)")

        if np.any(self.positions < 0.0) or np.any(self.positions > area_size):
            raise ValueError("initial positions must lie inside the simulation area")

        self.rng = rng
        self.max_speed = float(max_speed)
        self.preferred_speed = float(preferred_speed)
        self.rho_v = float(rho_v)
        self.sigma_v = float(sigma_v)
        self.area_size = float(area_size)
        self.dt = float(dt)
        self.refresh = int(pref_refresh_interval)
        self._step_count = 0

        n = len(self.positions)
        # Cumulative genuine boundary-reflection event counts, per object.
        self.boundary_event_counts = np.zeros(n, dtype=int)

        # Initial random velocity.
        angles = rng.uniform(0.0, 2.0 * np.pi, n)
        low = 0.25 * self.preferred_speed
        high = self.preferred_speed

        if high > 0.0:
            speeds = rng.uniform(low, high, n)
        else:
            speeds = np.zeros(n)

        speeds = np.minimum(speeds, self.max_speed)

        self.velocities = np.column_stack(
            [speeds * np.cos(angles), speeds * np.sin(angles)]
        )

        self._v_pref = self.velocities.copy()

    def _refresh_preferred(self):
        """Draw new preferred velocities."""
        n = len(self.positions)

        if self.preferred_speed <= 0.0:
            self._v_pref = np.zeros((n, 2), dtype=float)
            return

        angles = self.rng.uniform(0.0, 2.0 * np.pi, n)
        low = 0.25 * self.preferred_speed
        high = self.preferred_speed
        speeds = np.minimum(
            self.rng.uniform(low, high, n),
            self.max_speed,
        )

        self._v_pref = np.column_stack(
            [speeds * np.cos(angles), speeds * np.sin(angles)]
        )

    def step(self):
        """Advance by one time block and return position/velocity copies."""
        if self._step_count > 0 and self._step_count % self.refresh == 0:
            self._refresh_preferred()

        noise = (
            self.rng.standard_normal(self.velocities.shape) * self.sigma_v
        )

        self.velocities = (
            self.rho_v * self.velocities
            + (1.0 - self.rho_v) * self._v_pref
            + noise
        )

        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)

        mask = speeds > self.max_speed
        safe_speeds = np.maximum(speeds, np.finfo(float).eps)

        self.velocities = np.where(
            mask,
            self.velocities / safe_speeds * self.max_speed,
            self.velocities,
        )

        self.positions = self.positions + self.velocities * self.dt

        self.positions, self.velocities, boundary_hit = _reflect(
            self.positions,
            self.velocities,
            self.area_size,
        )
        self.boundary_event_counts += boundary_hit.astype(int)
        self._last_boundary_hit = boundary_hit

        self._step_count += 1

        return self.positions.copy(), self.velocities.copy()

    def last_boundary_hit(self):
        """
        Bool ndarray, shape (N,) — which objects genuinely reflected off a
        boundary on the most recent step() call. None before the first step.
        """
        return getattr(self, "_last_boundary_hit", None)
