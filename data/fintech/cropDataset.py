import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

# Load the massive 150MB original file
df = pd.read_csv(DATA_DIR / 'massive_creditcard.csv')

# Grab all 492 fraud cases (Class == 1)
frauds = df[df['Class'] == 1]

# Grab a random 10,000 normal cases (Class == 0)
normals = df[df['Class'] == 0].sample(n=10000, random_state=42)

# Combine and shuffle them
cropped_df = pd.concat([frauds, normals]).sample(frac=1, random_state=42)

# Save the new, tiny file to your project folder!
cropped_df.to_csv(DATA_DIR / 'original.csv', index=False)
