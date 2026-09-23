import yaml
from src.simulation.end_to_end import run_simulation
from unittest.mock import patch

def print_decision(*args, **kwargs):
    from src.optimization.churn_aware import churn_aware_decision as orig
    res = orig(*args, **kwargs)
    print(f"Pred Gain: {res['predicted_gain']:.2f}, Churn: {res['norm_churn_cost']:.2f}, Net: {res['net_gain']:.2f}")
    return res

with patch("src.simulation.end_to_end.churn_aware_decision", print_decision):
    res = run_simulation("PREDICTIVE+CHURN", time_blocks=10)
