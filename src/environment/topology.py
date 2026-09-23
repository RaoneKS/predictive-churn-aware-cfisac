import numpy as np


def generate_ap_positions(num_aps, area_size, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, area_size, size=(num_aps, 2))


def generate_positions(num_objects, area_size, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, area_size, size=(num_objects, 2))
