import json
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.stats import triang

# Load and simulate loss distributions from JSON configuration files
def load_config_and_simulate(directory: str, num_iterations: int = 10000) -> Dict[str, List[float]]:
    np.random.seed(42)
    all_results = {}

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)

            losses = []
            for _ in range(num_iterations):
                total_loss = 0
                for attack in data["attacks"]:
                    p_success = attack["lrs"]
                    u = np.random.uniform(0, 1)
                    if u < p_success:
                        a = attack["min"]
                        b = attack["mode"]
                        c = attack["max"]
                        c_scaled = c - a
                        mode_scaled = (b - a) / c_scaled if c_scaled != 0 else 0.5
                        loss = triang.rvs(c=mode_scaled, loc=a, scale=c_scaled)
                        total_loss += loss
                losses.append(total_loss)

            attack_type = data.get("name", filename)
            all_results[attack_type] = losses
    return all_results

# Generate and save the Loss Exceedance Curve chart
def plot_lec_and_save(simulated_losses: Dict[str, List[float]], output_dir: str = "output", filename: str = "lec_chart"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    for label, losses in simulated_losses.items():
        sorted_losses = np.sort(losses)
        exceedance_probs = 1.0 - np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
        plt.plot(sorted_losses, exceedance_probs, label=label)

    plt.xlabel("Loss ($)")
    plt.ylabel("Probability of Exceedance")
    plt.title("Loss Exceedance Curves")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    png_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()

    print(f"Saved PDF to: {pdf_path}")
    print(f"Saved PNG to: {png_path}")

# -----------------------
# Main execution block
# -----------------------
if __name__ == "__main__":
    input_dir = "lec_inputs"  # directory containing .json files
    output_dir = "output"     # directory where charts will be saved
    filename = "lec_chart"    # base name for output files

    #results = load_config_and_simulate(input_dir)
    #plot_lec_and_save(results, output_dir, filename)
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            full_path = os.path.join(input_dir, filename)
            with open(full_path, "r") as file:
                config = json.load(file)
            
            scenario_name = config.get("name", os.path.splitext(filename)[0])
            single_result = load_config_and_simulate(input_dir, num_iterations=10000)
            plot_lec_and_save(
                {scenario_name: single_result[scenario_name]},
                output_dir,
                filename=os.path.splitext(filename)[0]
            )

