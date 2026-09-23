import numpy as np
from src.optimization.churn_aware import evaluate_cluster_performance, churn_aware_decision

ap_positions = np.array([[0, 0], [100, 0], [0, 100], [100, 100]])
user_positions = np.array([[50, 50], [20, 20]])
target_positions = np.array([[60, 60]])

x_curr = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
y_tx_curr = np.array([[1], [0], [0], [0]])
y_rx_curr = np.array([[1], [0], [0], [0]])

perf_curr = evaluate_cluster_performance(x_curr, y_tx_curr, y_rx_curr, ap_positions, user_positions, target_positions)

print("Example KEEP:")
decision_keep = churn_aware_decision(x_curr, y_tx_curr, y_rx_curr, x_curr, y_tx_curr, y_rx_curr, perf_curr, perf_curr)
for k, v in decision_keep.items(): print(f"  {k}: {v}")

print("\nExample RECONFIGURE:")
x_pred = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
y_tx_pred = np.array([[0], [1], [0], [0]])
y_rx_pred = np.array([[0], [1], [0], [0]])

perf_pred = perf_curr.copy()
perf_pred["combined_objective"] += 50.0

decision_reconfig = churn_aware_decision(x_curr, y_tx_curr, y_rx_curr, x_pred, y_tx_pred, y_rx_pred, perf_curr, perf_pred, c_comm=1.0, c_sensing_tx=1.0, c_sensing_rx=1.0)
for k, v in decision_reconfig.items(): print(f"  {k}: {v}")

