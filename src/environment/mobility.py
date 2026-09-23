import numpy as np


def constant_velocity_step(position, velocity, dt):
    """Advance an object using a constant-velocity model."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    return position + velocity * dt


def predict_constant_velocity(
    position,
    velocity,
    horizon,
    dt,
):
    """
    Predict future positions:

        p(t+h) = p(t) + h*v*dt
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    return np.asarray([
        position + h * velocity * dt
        for h in range(1, horizon + 1)
    ])


def update_positions(
    positions,
    velocities,
    dt,
    area_size=None,
    return_events=False,
):
    """
    Update multiple moving objects.

    Optional boundary handling keeps objects inside
    the simulation area by reflecting their velocity.

    Parameters
    ----------
    return_events : bool
        If True, also return a bool ndarray (N,) that is True for objects
        that genuinely reflected off a boundary this step (a real
        geometric contact event, not merely a velocity-sign change).
        Default False preserves the original 2-tuple return signature for
        existing callers.
    """
    positions = np.asarray(positions, dtype=float).copy()
    velocities = np.asarray(velocities, dtype=float).copy()

    new_positions = positions + velocities * dt
    boundary_hit = np.zeros(len(new_positions), dtype=bool)

    if area_size is not None:
        for i in range(len(new_positions)):
            for d in range(2):
                if new_positions[i, d] < 0:
                    new_positions[i, d] = -new_positions[i, d]
                    velocities[i, d] *= -1
                    boundary_hit[i] = True

                elif new_positions[i, d] > area_size:
                    new_positions[i, d] = (
                        2 * area_size - new_positions[i, d]
                    )
                    velocities[i, d] *= -1
                    boundary_hit[i] = True

    if return_events:
        return new_positions, velocities, boundary_hit

    return new_positions, velocities


def generate_velocities(
    num_objects,
    max_speed,
    seed=42,
):
    """
    Generate random 2-D velocities.
    """
    rng = np.random.default_rng(seed)

    angles = rng.uniform(
        0,
        2 * np.pi,
        size=num_objects,
    )

    speeds = rng.uniform(
        0.25 * max_speed,
        max_speed,
        size=num_objects,
    )

    velocities = np.column_stack([
        speeds * np.cos(angles),
        speeds * np.sin(angles),
    ])

    return velocities
