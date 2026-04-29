import os
import pandas as pd

config_path = "exp/adult/ddpm_mlp_best/config.toml"
target_col = "income" 

print("Generating samples from TabDDPM...")
os.system(f"PYTHONPATH=. python scripts/pipeline.py --config {config_path} --sample")

# Locate the output file
exp_dir = "exp/adult/ddpm_mlp_best/"
possible_outputs = [os.path.join(exp_dir, "synthetic.csv"), os.path.join(exp_dir, "sample.csv")]

generated_file = next((path for path in possible_outputs if os.path.exists(path)), None)

if not generated_file:
    print("Error: Could not find the generated CSV. Did sampling fail?")
    exit(1)

df_synth = pd.read_csv(generated_file)

# Split into Minority and Majority
df_minority = df_synth[df_synth[target_col] == 1].copy()
df_majority = df_synth[df_synth[target_col] == 0].copy()

# Save directly into your central synthetic data folder
out_dir = "../data/adult/synthetic/"
os.makedirs(out_dir, exist_ok=True)

min_path = os.path.join(out_dir, "synthetic_minority_tabddpm.csv")
maj_path = os.path.join(out_dir, "synthetic_majority_tabddpm.csv")

df_minority.to_csv(min_path, index=False)
df_majority.to_csv(maj_path, index=False)

print(f"Success! Saved {len(df_minority)} minority rows to {min_path}")
print(f"Success! Saved {len(df_majority)} majority rows to {maj_path}")

# python sample_model.py

# PYTHONPATH=. python scripts/pipeline.py  --sample --config exp/adult/config.toml