import unittest

import numpy as np

from src.environment.stochastic_mobility import StochasticMobilityGroup
from src.simulation.mobility_simulator import MobilitySimulator


class TestStochasticMobility(unittest.TestCase):

    def setUp(self):
        self.positions = np.array(
            [
                [100.0, 100.0],
                [250.0, 300.0],
                [400.0, 200.0],
            ]
        )

    def test_reproducibility(self):
        a = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
        )
        b = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
        )

        for _ in range(30):
            pa, va = a.step()
            pb, vb = b.step()

            np.testing.assert_allclose(pa, pb)
            np.testing.assert_allclose(va, vb)

    def test_different_seeds_produce_different_paths(self):
        a = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
        )
        b = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(456),
        )

        for _ in range(20):
            a.step()
            b.step()

        self.assertFalse(np.allclose(a.positions, b.positions))

    def test_positions_remain_inside_area(self):
        model = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
            max_speed=15.0,
            sigma_v=5.0,
            area_size=500.0,
        )

        for _ in range(500):
            positions, _ = model.step()

            self.assertTrue(np.all(positions >= 0.0))
            self.assertTrue(np.all(positions <= 500.0))

    def test_speed_limit(self):
        model = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
            max_speed=7.0,
            preferred_speed=12.0,
            sigma_v=8.0,
        )

        for _ in range(200):
            _, velocities = model.step()
            speeds = np.linalg.norm(velocities, axis=1)

            self.assertTrue(np.all(speeds <= 7.0 + 1e-10))

    def test_trajectory_is_not_constant_velocity(self):
        model = StochasticMobilityGroup(
            self.positions,
            np.random.default_rng(123),
            sigma_v=2.0,
        )

        velocity_history = []

        for _ in range(30):
            _, velocities = model.step()
            velocity_history.append(velocities.copy())

        velocity_history = np.asarray(velocity_history)
        deltas = np.diff(velocity_history, axis=0)

        self.assertGreater(
            np.max(np.linalg.norm(deltas, axis=2)),
            1e-8,
        )

    def test_simulator_history_shape(self):
        simulator = MobilitySimulator(
            user_positions=self.positions[:2],
            target_positions=self.positions[2:],
            seed=42,
            mobility_config={"type": "stochastic"},
            history_capacity=20,
        )

        for _ in range(10):
            simulator.step()

        users = simulator.get_user_history(10)
        targets = simulator.get_target_history(10)

        self.assertEqual(users.shape, (2, 10, 2))
        self.assertEqual(targets.shape, (1, 10, 2))

    def test_boundary_event_indicator_is_genuine(self):
        """
        boundary_event_counts must only increase when a position actually
        contacts and reflects off the simulation boundary -- not merely
        because stochastic noise flipped a velocity component's sign.
        """
        # A tightly confined, low-speed, low-noise model should essentially
        # never reach the boundary, so the genuine event count should stay
        # at (or very near) zero even though velocity components will
        # still flip sign frequently due to noise.
        model = StochasticMobilityGroup(
            np.array([[250.0, 250.0]]),   # centre of a 500x500 area
            np.random.default_rng(7),
            max_speed=2.0,
            preferred_speed=1.0,
            rho_v=0.9,
            sigma_v=0.1,
            area_size=500.0,
        )

        sign_flips = 0
        prev_vel = model.velocities.copy()

        for _ in range(50):
            _, vel = model.step()
            flipped = np.any(np.sign(vel) != np.sign(prev_vel))
            if flipped:
                sign_flips += 1
            prev_vel = vel.copy()

        self.assertEqual(
            model.boundary_event_counts[0], 0,
            "object never approached the boundary; genuine event count must be zero",
        )
        # Sanity: sign flips are a separate, noisier phenomenon and are not
        # asserted to be zero -- this is exactly the distinction the metric
        # fix is meant to preserve.

    def test_boundary_event_indicator_fires_at_boundary(self):
        """
        An object pushed hard against a wall with high speed must
        eventually register a genuine boundary_event_counts increment.
        """
        model = StochasticMobilityGroup(
            np.array([[1.0, 250.0]]),   # right next to the x=0 wall
            np.random.default_rng(11),
            max_speed=20.0,
            preferred_speed=20.0,
            rho_v=0.99,
            sigma_v=0.0,
            area_size=500.0,
        )
        # Force velocity toward the wall.
        model.velocities[:] = np.array([[-20.0, 0.0]])
        model._v_pref[:] = np.array([[-20.0, 0.0]])

        model.step()

        self.assertGreater(
            model.boundary_event_counts[0], 0,
            "object driven directly into a wall must register a genuine boundary event",
        )

    def test_constant_velocity_mode_remains_available(self):
        simulator = MobilitySimulator(
            user_positions=self.positions[:2],
            target_positions=self.positions[2:],
            seed=42,
        )

        initial = simulator.user_velocities.copy()

        for _ in range(20):
            simulator.step()

        # Constant-velocity mobility preserves speed. A boundary reflection
        # may flip the sign of an affected velocity component.
        np.testing.assert_allclose(
            np.abs(simulator.user_velocities),
            np.abs(initial),
        )


if __name__ == "__main__":
    unittest.main()
