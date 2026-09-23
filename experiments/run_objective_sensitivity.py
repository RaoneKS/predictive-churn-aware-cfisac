import os
import pandas as pd
from src.simulation.end_to_end import run_simulation

def main():
    out_dir = "results/objective_sensitivity"
    os.makedirs(out_dir, exist_ok=True)
    
    # Predefined sensitivity cases designed to explicitly change KEEP/RECONFIGURE bounds
    # By modifying objective weights, the relative importance of gain vs churn shifts.
    cases = {
        "baseline": {}, # Default config
        "high_churn_penalty": {
            "lambda_churn": 10.0,
        },
        "low_churn_penalty": {
            "lambda_churn": 0.1,
        },
        "zero_churn_penalty": {
            "lambda_churn": 0.0,
        },
        "high_comm_reward": {
            "alpha_communication": 10.0,
        },
    }
    
    summary_metrics = []
    
    for case_name, overrides in cases.items():
        print(f"Running PREDICTIVE+CHURN for sensitivity case: {case_name}")
        
        # We exclusively test PREDICTIVE+CHURN across different weights
        metrics = run_simulation(
            policy_name="PREDICTIVE+CHURN",
            config_path="configs/default.yaml",
            seed=42, # SAME mobility realization
            **overrides
        )
        
        summ = {
            "case": case_name,
            "avg_user_rate": metrics["avg_user_rate"].mean(),
            "p5_user_rate": metrics["p5_user_rate"].mean(),
            "sensing_utility": metrics["sensing_utility"].mean(),
            "energy": metrics["energy"].mean(),
            "fronthaul": metrics["fronthaul"].mean(),
            "cumulative_churn": metrics["cumulative_churn"][-1],
            "total_reconfigurations": metrics["reconfigurations"].sum(),
            "keep_count": metrics["keep_decisions"],
            "reconfigure_count": metrics["reconfigure_decisions"],
            "combined_objective": metrics["combined_objective"].mean(),
        }
        summary_metrics.append(summ)

    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(f"{out_dir}/sensitivity_summary.csv", index=False)
    
    print("\n" + "="*90)
    print("OBJECTIVE SENSITIVITY SUMMARY (PREDICTIVE+CHURN)")
    print("="*90)
    print(summary_df.to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    main()
