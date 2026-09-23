"""
src/environment/mobility_configs.py

Single source of truth for the stochastic-mobility regime parameters used
by BOTH experiments/validate_mobility.py and experiments/compare_mobility.py.

Keeping this in one place guarantees the mobility-validation report and the
2x2 mobility/predictor experiment describe and evaluate the exact same
stochastic regime -- no risk of the two scripts silently drifting apart.
"""

STOCHASTIC_CONFIG = {
    "type": "stochastic",
    "users": {
        "max_speed": 15.0,
        "preferred_speed": 8.0,
        "rho_v": 0.85,
        "sigma_v": 2.0,
        "pref_refresh_interval": 20,
    },
    "targets": {
        "max_speed": 10.0,
        "preferred_speed": 5.0,
        "rho_v": 0.75,
        "sigma_v": 3.0,
        "pref_refresh_interval": 20,
    },
}
