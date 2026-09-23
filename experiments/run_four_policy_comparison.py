import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.simulation.end_to_end import run_simulation

def main():
    out_dir = "results/four_policy_comparison"
    os.makedirs(out_dir, exist_ok=True)
    
    policies = ["STATIC", "REACTIVE", "PREDICTIVE", "PREDICTIVE+CHURN"]
    
    results = {}
    for p in policies:
        print(f"Running simulation for {p}...")
        results[p] = run_simulation(
            policy_name=p,
            config_path="configs/default.yaml",
            seed=42, # Same seed across policies for fairness
        )
        
    all_metrics = []
    summary_metrics = []
    
    for p in policies:
        df = pd.DataFrame({k: v for k, v in results[p].items() if isinstance(v, np.ndarray)})
        df["policy"] = p
        df["block"] = np.arange(100)
        all_metrics.append(df)
        
        summ = {
            "policy": p,
            "avg_user_rate_mean": df["avg_user_rate"].mean(),
            "p5_user_rate_mean": df["p5_user_rate"].mean(),
            "sensing_utility_mean": df["sensing_utility"].mean(),
            "energy_mean": df["energy"].mean(),
            "churn_mean": df["churn"].mean(),
            "cumulative_churn_total": df["cumulative_churn"].iloc[-1],
            "combined_objective_mean": df["combined_objective"].mean(),
            "total_reconfigurations": df["reconfigurations"].sum(),
            "keep_decisions": results[p].get("keep_decisions", 0),
            "reconfigure_decisions": results[p].get("reconfigure_decisions", 0)
        }
        summary_metrics.append(summ)

    full_df = pd.concat(all_metrics, ignore_index=True)
    full_df.to_csv(f"{out_dir}/block_metrics.csv", index=False)
    
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(f"{out_dir}/summary_metrics.csv", index=False)
    
    metrics_to_plot = [
        ("avg_user_rate", "Average User Rate vs Time"),
        ("p5_user_rate", "5th Percentile User Rate vs Time"),
        ("sensing_utility", "Sensing Information Gain vs Time"),
        ("tracking_violations", "Tracking Error vs Time"),
        ("energy", "Energy vs Time"),
        ("fronthaul", "Fronthaul Load vs Time"),
        ("churn", "Churn vs Time"),
        ("cumulative_churn", "Cumulative Churn vs Time"),
        ("combined_objective", "Combined Objective vs Time"),
        ("reconfigurations", "Reconfiguration Count vs Time")
    ]
    
    for metric_key, title in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for p in policies:
            plt.plot(results[p][metric_key], label=p)
        plt.title(title)
        plt.xlabel("Time Block")
        plt.ylabel(metric_key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{metric_key}.png")
        plt.close()
        
    print("\n" + "="*80)
    print("FOUR-POLICY COMPARISON SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    print("Normalized Objective Equation:")
    print("combined_objective = (alpha_comm * raw_comm * comm_scale) + (beta_sens * raw_sens * sens_scale)")
    print("                     - lambda_energy * raw_energy * energy_scale")
    print("                     - lambda_fronthaul * raw_fronthaul * fronthaul_scale")
    print("                     - lambda_churn * raw_churn * churn_scale")
    print("                     - [QoS/Tracking violations similarly scaled]")

if __name__ == "__main__":
    main()
