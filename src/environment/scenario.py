from dataclasses import dataclass


@dataclass
class Scenario:
    num_aps: int = 20
    num_users: int = 5
    num_targets: int = 2
    horizon: int = 5
    area_size: float = 500.0
    dt: float = 1.0
