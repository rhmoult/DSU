import os
import json
import csv

# Path to your input and output folders
INPUT_DIR = "./lec_inputs"
OUTPUT_FILE = "./output/simulated_expected_loss.csv"

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def triangle_expected_loss(min_val, mode_val, max_val):
    return (min_val + mode_val + max_val) / 3.0

# Collect results
results = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
            experiment_name = data.get("name", filename.replace(".json", ""))
            for attack in data.get("attacks", []):
                attack_name = attack.get("name")
                lrs = attack.get("lrs", 1.0)
                min_val = attack.get("min", 0)
                mode_val = attack.get("mode", 0)
                max_val = attack.get("max", 0)

                expected_loss = triangle_expected_loss(min_val, mode_val, max_val)
                adjusted_loss = expected_loss * lrs

                results.append({
                    "Experiment": experiment_name,
                    "Attack Type": attack_name,
                    "LRS": lrs,
                    "Min": min_val,
                    "Mode": mode_val,
                    "Max": max_val,
                    "Expected Loss": round(expected_loss, 2),
                    "Simulated Expected Loss": round(adjusted_loss, 2),
                })

# Write to CSV
with open(OUTPUT_FILE, "w", newline="") as csvfile:
    fieldnames = [
        "Experiment", "Attack Type", "LRS", "Min", "Mode", "Max",
        "Expected Loss", "Simulated Expected Loss"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Saved simulated expected losses to {OUTPUT_FILE}")

